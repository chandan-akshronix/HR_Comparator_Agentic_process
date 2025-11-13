import sys, os, logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import main_workflow
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.output_parsers import JsonOutputParser
import json
from langgraph_flow import build_langgraph
from orchestrator import compute_stability_score, postprocess_recruiter_logic

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()
logging.basicConfig(level=logging.INFO)

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hr_resume_comparator")

client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

app = FastAPI(title="AI Resume–JD Matching Backend")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite frontend
    "http://localhost:8000"   # Main backend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Request/Response Models
# ============================================
class BatchComparisonRequest(BaseModel):
    workflow_id: str
    jd_text: str
    resumes: List[Dict[str, str]]  # [{resume_id, resume_text}, ...]

# ============================================
# NEW: Batch Comparison Endpoint
# ============================================
@app.post("/compare-batch")
def compare_batch(request: BatchComparisonRequest):
    """
    Batch comparison endpoint for main backend integration
    Processes multiple resumes against one JD using LangGraph
    """
    try:
        results = []
        parser = JsonOutputParser()
        graph = build_langgraph()
        app_graph = graph.compile()
        
        logging.info(f"🤖 Processing {len(request.resumes)} resumes for workflow {request.workflow_id}")
        
        for idx, resume in enumerate(request.resumes, 1):
            resume_id = resume["resume_id"]
            resume_text = resume["resume_text"]
            
            logging.info(f"▶️ Processing resume {idx}/{len(request.resumes)}: {resume_id}")
            
            # Run LangGraph pipeline
            state = {"jd_text": request.jd_text, "resume_text": resume_text}
            result_state = app_graph.invoke(state)
            
            # Parse comparison result
            comparison_result = result_state.get("comparison_result", "")
            try:
                comp_json = parser.parse(comparison_result)
            except Exception as e:
                logging.warning(f"⚠️ Failed to parse comparison result: {e}")
                comp_json = {
                    "fit_category": "Unknown",
                    "total_score": 0,
                    "selection_reason": comparison_result[:500],
                    "parameter_breakdown": {},
                    "risk_factors": [],
                    "growth_signals": []
                }
            
            # Extract resume data
            resume_raw = result_state.get("resume_extracted", "")
            try:
                resume_data = json.loads(
                    resume_raw.replace("```json", "").replace("```", "").strip()
                )
            except Exception as e:
                logging.warning(f"⚠️ Failed to parse resume data: {e}")
                resume_data = {}
            
            # Compute stability score
            stability_score, gap_flags = compute_stability_score(
                resume_data.get("Career_History", [])
            )
            comp_json["stability_score"] = stability_score
            
            # Merge risk factors
            existing_risks = comp_json.get("risk_factors", [])
            if isinstance(existing_risks, list):
                comp_json["risk_factors"] = list(set(existing_risks + gap_flags))
            else:
                comp_json["risk_factors"] = gap_flags
            
            # Apply recruiter logic
            comp_json = postprocess_recruiter_logic(comp_json)
            
            # Extract JD data
            jd_raw = result_state.get("jd_extracted", "")
            try:
                jd_data = json.loads(
                    jd_raw.replace("```json", "").replace("```", "").strip()
                )
            except:
                jd_data = {}
            
            # Format result for main backend
            result = {
                "resume_id": resume_id,
                "match_score": float(comp_json.get("total_score", 0)),
                "fit_category": comp_json.get("fit_category", "Unknown"),
                "jd_extracted": jd_data,
                "resume_extracted": resume_data,
                "match_breakdown": comp_json.get("parameter_breakdown", {}),
                "selection_reason": comp_json.get("selection_reason", ""),
                "confidence_score": comp_json.get("recruiter_confidence", "Low"),
                "risk_factors": comp_json.get("risk_factors", []),
                "growth_signals": comp_json.get("growth_signals", []),
                "stability_score": stability_score
            }
            
            results.append(result)
            logging.info(f"✅ Completed resume {idx}: Score={result['match_score']}, Fit={result['fit_category']}")
        
        response = {
            "workflow_id": request.workflow_id,
            "total_resumes": len(results),
            "processing_time_ms": 2500,  # Can calculate actual time if needed
            "results": results
        }
        
        logging.info(f"✅ Batch processing complete for workflow {request.workflow_id}")
        return response
        
    except Exception as e:
        logging.error(f"❌ Batch comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "AI Resume Comparator Agent"
    }

@app.post("/run_workflow/{workflow_id}")
def run_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(main_workflow, workflow_id)
        logging.info(f"🚀 Workflow {workflow_id} started.")
        return {"message": f"Workflow {workflow_id} started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/status/{workflow_id}")
def get_workflow_status(workflow_id: str):
    try:
        wf = db["hr_resume_comparator.workflow_executions"].find_one({"workflow_id": workflow_id}, {"_id": 0})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"status": wf.get("status"), "progress": wf.get("progress", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "AI Workflow Backend Connected ✅"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=9000, reload=True)
