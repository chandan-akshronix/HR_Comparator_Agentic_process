import os, uuid, json, logging
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from mongo_input import (
    fetch_jd_by_id,
    fetch_resumes_by_jd_id,
    INPUT_MONGO_URI,
    INPUT_MONGO_DB,
    INPUT_MONGO_COLLECTION,
    INPUT_MONGO_JD_COLLECTION,
)
from mongo_helper import store_to_mongo
from file_helper import store_to_file
from langgraph_flow import build_langgraph
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ============================================================
# Stability score (same as before)
# ============================================================
def compute_stability_score(career_history):
    # [same implementation as your existing version]
    if not career_history or not isinstance(career_history, list):
        return 50, ["Career history incomplete or missing"]
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
    if job_count == 0:
        return 50, ["Career data not parseable"]
    parsed_jobs.sort(key=lambda x: x[0])
    for i in range(1, len(parsed_jobs)):
        prev_end = parsed_jobs[i-1][1]
        curr_start = parsed_jobs[i][0]
        gap_years = curr_start - prev_end
        if gap_years > 0:
            gap_penalty += gap_years * 5
            risk_flags.append(f"Employment gap of {gap_years} year(s)")
    avg_tenure = total_duration / job_count
    score = 50
    if avg_tenure >= 3: score += 20
    elif 1.5 <= avg_tenure < 3: score += 10
    elif avg_tenure < 1:
        score -= 20
        risk_flags.append("Frequent job changes (<1 year avg)")
    score -= gap_penalty
    if len(career_history) >= 2:
        latest_title = career_history[-1].get("Job_Title", "").lower()
        if "lead" in latest_title or "manager" in latest_title: score += 10
        elif "intern" in latest_title or "trainee" in latest_title: score -= 5
    score = max(0, min(score, 100))
    return score, risk_flags

# ============================================================
# Recruiter post-processing (same as before)
# ============================================================
def postprocess_recruiter_logic(comp_json):
    score = int(comp_json.get("total_score", 0))
    risks = comp_json.get("risk_factors", [])
    growths = comp_json.get("growth_signals", [])
    if len(risks) >= 3: score -= 10
    elif len(risks) == 2: score -= 5
    if len(growths) >= 2: score += 5
    elif len(growths) >= 3: score += 10
    score = max(0, min(score, 100))
    if score >= 85: fit, confidence = "Best Fit", "High"
    elif 60 <= score < 85: fit, confidence = "Partial Fit", "Medium"
    else: fit, confidence = "Not Fit", "Low"
    comp_json["total_score"] = score
    comp_json["fit_category"] = fit
    comp_json["recruiter_confidence"] = confidence
    return comp_json

# ============================================================
# Helpers to update job/resume status
# ============================================================
def set_job_status(jd_id, status):
    """Update job_descriptions.job_status field."""
    try:
        client = MongoClient(INPUT_MONGO_URI)
        db = client[INPUT_MONGO_DB]
        db[INPUT_MONGO_JD_COLLECTION].update_one(
            {"_id": jd_id},
            {"$set": {
                "job_status": status,
                "last_updated": datetime.utcnow().isoformat()
            }},
            upsert=False,
        )
        client.close()
        logging.info(f"ℹ️ Job {jd_id} status set to '{status}'.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to set job status: {e}")

def mark_resume_processed(resume_doc_id):
    """Mark a resume as processed=True in input DB."""
    try:
        client = MongoClient(INPUT_MONGO_URI)
        db = client[INPUT_MONGO_DB]
        db[INPUT_MONGO_COLLECTION].update_one(
            {"_id": ObjectId(resume_doc_id)},
            {"$set": {"processed": True, "processed_at": datetime.utcnow().isoformat()}},
        )
        client.close()
        logging.info(f"🗂️ Marked resume {resume_doc_id} as processed.")
    except Exception as e:
        logging.warning(f"⚠️ Could not mark resume processed: {e}")

# ============================================================
# 🚀 MAIN PIPELINE (patched for jd_id-based processing)
# ============================================================
def main(jd_id=None):
    if not jd_id:
        logging.error("❌ JD ID not provided. Cannot run agent.")
        return

    set_job_status(jd_id, "processing")

    MONGO_URI = os.getenv("MONGO_URI")
    result_client = MongoClient(MONGO_URI)
    result_db = result_client["resume_selector"]
    result_collection = result_db["selected_resumes"]

    jd_text = fetch_jd_by_id(jd_id)
    resumes = fetch_resumes_by_jd_id(jd_id)
    if not jd_text or not resumes:
        logging.error(f"❌ No JD or unprocessed resumes found for JD {jd_id}.")
        set_job_status(jd_id, "no_data")
        return

    graph = build_langgraph()
    app = graph.compile()

    for r in resumes:
        resume_text = r["text"]
        resume_doc_id = r.get("_id")
        resume_id = str(uuid.uuid4())
        logging.info(f"▶️ Processing resume {resume_doc_id} for JD {jd_id}")

        # LangGraph LLM flow
        state = {"jd_text": jd_text, "resume_text": resume_text}
        result_state = app.invoke(state)
        comparison_result = result_state.get("comparison_result", "")

        # Parse comparator JSON
        try:
            parser = JsonOutputParser()
            comp_json = parser.parse(comparison_result)
        except Exception:
            comp_json = {
                "fit_category": "Unknown",
                "total_score": 0,
                "selection_reason": comparison_result,
            }

        # Parse resume JSON
        resume_raw = result_state.get("resume_extracted", "")
        try:
            resume_data = json.loads(resume_raw.replace("```json", "").replace("```", "").strip())
        except Exception:
            resume_data = {}

        # Stability score
        stability_score, gap_flags = compute_stability_score(resume_data.get("Career_History", []))
        existing_risks = comp_json.get("risk_factors", [])
        comp_json["risk_factors"] = list(set(existing_risks + gap_flags))

        comp_json = postprocess_recruiter_logic(comp_json)

        # Add metadata
        comp_json.update({
            "jd_id": jd_id,
            "resume_id": resume_id,
            "applicant_name": resume_data.get("Name", ""),
            "applicant_email": resume_data.get("Email", ""),
            "applicant_mobile": resume_data.get("Mobile", ""),
            "skill_score": comp_json.get("parameter_breakdown", {}).get("Skill_Score", 0),
            "experience_score": comp_json.get("parameter_breakdown", {}).get("Experience_Score", 0),
            "stability_score": stability_score,
            "full_resume_data": resume_data,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Save results
        store_to_file(comp_json, resume_id)
        store_to_mongo(comp_json, resume_id, result_collection)
        mark_resume_processed(resume_doc_id)

    set_job_status(jd_id, "completed")
    result_client.close()
    logging.info(f"🏁 Completed matching for JD {jd_id}")
