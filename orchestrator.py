import os, json, logging
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from langchain_core.output_parsers import JsonOutputParser
from mongo_input import (
    fetch_jd_by_id,
    fetch_resume_by_id,
    fetch_workflow_by_id,
)
from mongo_helper import store_to_mongo
from file_helper import store_to_file
from langgraph_flow import build_langgraph

load_dotenv()
logging.basicConfig(level=logging.INFO)


# ============================================================
# Stability Score
# ============================================================
def compute_stability_score(career_history):
    # If no career history or empty list, stability cannot be judged - return 0
    if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
        return 0, ["No professional experience - stability cannot be assessed"]
    total_duration = 0
    job_count = 0
    gap_penalty = 0
    risk_flags = []
    parsed_jobs = []
    for job in career_history:
        start = job.get("Start_Date", "")
        end = job.get("End_Date", "")
        if not start:
            continue
        try:
            start_year = int("".join([ch for ch in start if ch.isdigit()])[:4])
            end_year = int("".join([ch for ch in end if ch.isdigit()])[:4]) if end else datetime.now().year
            parsed_jobs.append((start_year, end_year))
            total_duration += max(0, end_year - start_year)
            job_count += 1
        except Exception:
            continue
    # If no valid jobs parsed, stability cannot be judged - return 0
    if job_count == 0:
        return 0, ["No valid professional experience found - stability cannot be assessed"]
    parsed_jobs.sort(key=lambda x: x[0])
    for i in range(1, len(parsed_jobs)):
        prev_end = parsed_jobs[i - 1][1]
        curr_start = parsed_jobs[i][0]
        gap_years = curr_start - prev_end
        if gap_years > 0:
            gap_penalty += gap_years * 5
            risk_flags.append(f"Employment gap of {gap_years} year(s)")
    avg_tenure = total_duration / job_count
    score = 50
    if avg_tenure >= 3:
        score += 20
    elif 1.5 <= avg_tenure < 3:
        score += 10
    elif avg_tenure < 1:
        score -= 20
        risk_flags.append("Frequent job changes (<1 year avg)")
    score -= gap_penalty
    if len(career_history) >= 2:
        latest_title = career_history[-1].get("Job_Title", "").lower()
        if "lead" in latest_title or "manager" in latest_title:
            score += 10
        elif "intern" in latest_title or "trainee" in latest_title:
            score -= 5
    score = max(0, min(score, 100))
    return score, risk_flags


# ============================================================
# Recruiter Postprocessing
# ============================================================
def postprocess_recruiter_logic(comp_json):
    score = int(comp_json.get("total_score", 0))
    risks = comp_json.get("risk_factors", [])
    growths = comp_json.get("growth_signals", [])
    if len(risks) >= 3:
        score -= 10
    elif len(risks) == 2:
        score -= 5
    if len(growths) >= 2:
        score += 5
    elif len(growths) >= 3:
        score += 10
    score = max(0, min(score, 100))
    if score >= 85:
        fit, confidence = "Best Fit", "High"
    elif 60 <= score < 85:
        fit, confidence = "Partial Fit", "Medium"
    else:
        fit, confidence = "Not Fit", "Low"
    comp_json["total_score"] = score
    comp_json["fit_category"] = fit
    comp_json["recruiter_confidence"] = confidence
    return comp_json


# ============================================================
# Workflow-based pipeline
# ============================================================
def main_workflow(workflow_id=None):
    if not workflow_id:
        logging.error("❌ Workflow ID not provided.")
        return

    client = MongoClient(os.getenv("MONGO_URI"))
    wf_db = client["hr_resume_comparator"]
    wf_collection = wf_db["hr_resume_comparator.workflow_executions"]

    workflow_doc = wf_collection.find_one({"workflow_id": workflow_id})
    if not workflow_doc:
        logging.error(f"❌ Workflow {workflow_id} not found.")
        client.close()
        return

    jd_id = workflow_doc.get("jd_id")
    resume_ids = workflow_doc.get("resume_ids", [])
    total_resumes = len(resume_ids)
    processed_count = 0

    if not jd_id or not resume_ids:
        logging.error(f"⚠️ Missing jd_id or resume_ids in workflow {workflow_id}.")
        client.close()
        return

    logging.info(f"🚀 Running workflow {workflow_id} for JD {jd_id} ({total_resumes} resumes)")

    jd_text = fetch_jd_by_id(jd_id)
    if not jd_text:
        logging.error(f"❌ JD text not found for {jd_id}.")
        client.close()
        return

    graph = build_langgraph()
    app = graph.compile()

    result_client = MongoClient(os.getenv("MONGO_URI"))
    result_collection = result_client["resume_selector"]["selected_resumes"]

    for resume_oid in resume_ids:
        try:
            resume_id = str(resume_oid)
            logging.info(f"▶️ Processing resume {resume_id}")

            resume_text = fetch_resume_by_id(ObjectId(resume_oid))
            if not resume_text:
                logging.warning(f"⚠️ Resume not found for ID {resume_id}")
                continue

            state = {"jd_text": jd_text, "resume_text": resume_text}
            result_state = app.invoke(state)
            comparison_result = result_state.get("comparison_result", "")

            parser = JsonOutputParser()
            try:
                comp_json = parser.parse(comparison_result)
            except Exception:
                comp_json = {"fit_category": "Unknown", "total_score": 0, "selection_reason": comparison_result}

            resume_raw = result_state.get("resume_extracted", "")
            try:
                resume_data = json.loads(resume_raw.replace("```json", "").replace("```", "").strip())
            except Exception:
                resume_data = {}

            stability_score, gap_flags = compute_stability_score(resume_data.get("Career_History", []))
            comp_json["stability_score"] = stability_score
            comp_json["risk_factors"] = list(set(comp_json.get("risk_factors", []) + gap_flags))
            comp_json = postprocess_recruiter_logic(comp_json)

            comp_json.update({
                "workflow_id": workflow_id,
                "jd_id": jd_id,
                "resume_id": resume_id,
                "applicant_name": resume_data.get("Name", ""),
                "applicant_email": resume_data.get("Email", ""),
                "applicant_mobile": resume_data.get("Mobile", ""),
                "full_resume_data": resume_data,
                "timestamp": datetime.utcnow().isoformat()
            })

            store_to_file(comp_json, resume_id)
            store_to_mongo(comp_json, resume_id, result_collection)

            processed_count += 1
            wf_collection.update_one(
                {"workflow_id": workflow_id},
                {"$set": {
                    "processed_resumes": processed_count,
                    "progress.percentage": int((processed_count / total_resumes) * 100),
                    "progress.completed_agents": processed_count
                }}
            )
        except Exception as e:
            logging.error(f"❌ Error processing resume {resume_oid}: {e}")
            continue

    wf_collection.update_one(
        {"workflow_id": workflow_id},
        {"$set": {"status": "completed", "completed_at": datetime.utcnow().isoformat()}}
    )

    client.close()
    result_client.close()
    logging.info(f"🏁 Completed workflow {workflow_id} ({processed_count}/{total_resumes})")
