import sys, os, logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
DATABASE_NAME = os.getenv("DATABASE_NAME", "pod_1")


def resolve_cors_origins() -> tuple[List[str], bool]:
    """
    Returns a tuple of (origins, allow_credentials) based on CORS_ORIGINS env.
    If no origins are provided, fall back to wildcard origins without credentials.
    """
    raw_origins = os.getenv("CORS_ORIGINS", "")
    parsed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if parsed_origins:
        return parsed_origins, True
    return ["*"], False

# Connection options for MongoDB Atlas
def get_connection_options():
    if not MONGODB_URL:
        return {}
    if "mongodb+srv://" in MONGODB_URL:
        # SRV connections automatically use TLS
        return {
            "serverSelectionTimeoutMS": 30000,
            "connectTimeoutMS": 20000,
            "socketTimeoutMS": 20000,
            "retryWrites": True,
        }
    elif "mongodb://" in MONGODB_URL and "mongodb.net" in MONGODB_URL:
        # Standard Atlas connections need explicit TLS
        return {
            "tls": True,
            "tlsAllowInvalidCertificates": False,
            "tlsCAFile": None,  # Use system CA certificates
            "serverSelectionTimeoutMS": 30000,
            "connectTimeoutMS": 20000,
            "socketTimeoutMS": 20000,
            "retryWrites": True,
        }
    return {}

client = MongoClient(MONGODB_URL, **get_connection_options())
db = client[DATABASE_NAME]

app = FastAPI(title="AI Resume–JD Matching Backend")

cors_origins, allow_credentials = resolve_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================
# Prometheus Metrics Instrumentation
# ============================================
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
)

instrumentator.instrument(app).expose(app, endpoint="/metrics")

# ============================================
# Request/Response Models
# ============================================
class BatchComparisonRequest(BaseModel):
    workflow_id: str
    jd_id: str  # Added jd_id for saving results
    jd_text: str
    resumes: List[Dict[str, str]]  # [{resume_id, resume_text}, ...]

# ============================================
# Helper: Process Single Resume (for parallel execution)
# ============================================
def process_single_resume_sequential(
    resume_id: str,
    resume_text: str,
    jd_text: str,
    app_graph,
    parser: JsonOutputParser,
    idx: int,
    total: int,
    workflow_id: str = None,
    jd_id: str = None
) -> dict:
    """
    Process a single resume through full 3-agent pipeline
    This function runs in parallel with other resumes
    """
    resume_start = time.time()
    
    logging.info(f"▶️ [{idx}/{total}] Processing resume: {resume_id}")
    
    try:
        # Run full 3-agent pipeline: JD Extractor → Resume Extractor → Comparator
        state = {"jd_text": jd_text, "resume_text": resume_text}
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
        
        # Format result
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
        
        resume_time = time.time() - resume_start
        logging.info(f"✅ [{idx}/{total}] Completed: {resume_id} - Score={result['match_score']}, Fit={result['fit_category']} ({resume_time:.1f}s)")
        
        # Save result immediately to database for live updates
        if workflow_id and jd_id:
            try:
                from bson import ObjectId
                from datetime import datetime
                
                # Delete existing result if any
                existing = db["resume_result"].find_one({
                    "resume_id": ObjectId(resume_id) if isinstance(resume_id, str) and len(resume_id) == 24 else resume_id,
                    "jd_id": jd_id
                })
                if existing:
                    db["resume_result"].delete_one({"_id": existing["_id"]})
                
                # Insert new result
                result_doc = {
                    "resume_id": ObjectId(resume_id) if isinstance(resume_id, str) and len(resume_id) == 24 else resume_id,
                    "jd_id": jd_id,
                    "workflow_id": workflow_id,
                    "match_score": result.get("match_score", 0),
                    "fit_category": result.get("fit_category", "Unknown"),
                    "jd_extracted": result.get("jd_extracted", {}),
                    "resume_extracted": result.get("resume_extracted", {}),
                    "match_breakdown": result.get("match_breakdown", {}),
                    "selection_reason": result.get("selection_reason", ""),
                    "agent_version": "v1.0.0",
                    "confidence_score": result.get("confidence_score", "Unknown"),
                    "timestamp": datetime.utcnow()
                }
                db["resume_result"].insert_one(result_doc)
                
                # Update workflow metrics incrementally
                current_matches = list(db["resume_result"].find({
                    "workflow_id": workflow_id,
                    "jd_id": jd_id
                }))
                high_matches_count = len([m for m in current_matches if m.get("match_score", 0) >= 80])
                
                # Update workflow execution document
                db["workflow_executions"].update_one(
                    {"workflow_id": workflow_id},
                    {
                        "$set": {
                            "metrics.candidates_scored": len(current_matches),
                            "metrics.high_matches": high_matches_count,
                            "metrics.best_fit": high_matches_count,
                            "metrics.partial_fit": len([m for m in current_matches if 50 <= m.get("match_score", 0) < 80]),
                            "metrics.not_fit": len([m for m in current_matches if m.get("match_score", 0) < 50]),
                            "progress.processed_count": len(current_matches),
                            "progress.percentage": int((len(current_matches) / total) * 100) if total > 0 else 0
                        }
                    }
                )
                
                logging.info(f"💾 [{idx}/{total}] Saved result to database for live updates - {len(current_matches)}/{total} processed")
            except Exception as save_error:
                logging.warning(f"⚠️ Failed to save result for live update: {save_error}")
                import traceback
                traceback.print_exc()
        
        return result
        
    except Exception as e:
        logging.error(f"❌ Error processing resume {resume_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "resume_id": resume_id,
            "match_score": 0,
            "fit_category": "Error",
            "jd_extracted": {},
            "resume_extracted": {},
            "match_breakdown": {},
            "selection_reason": f"Processing error: {str(e)}",
            "confidence_score": "Low",
            "risk_factors": ["Processing Error"],
            "growth_signals": [],
            "stability_score": 0
        }

# ============================================
# Batch Comparison Endpoint (5 Parallel Workers)
# ============================================
@app.post("/compare-batch")
async def compare_batch(request: BatchComparisonRequest):
    """
    PARALLEL PROCESSING: Processes 5 resumes simultaneously
    
    Each resume goes through 3 agents sequentially:
    1. JD Extractor → 2. Resume Extractor → 3. Comparator
    
    But 5 resumes run in parallel for 5x speed improvement!
    - 24 resumes: ~6-7 batches × 30s = ~3-4 minutes (vs 12 minutes sequential)
    """
    try:
        start_time = time.time()
        parser = JsonOutputParser()
        graph = build_langgraph()
        app_graph = graph.compile()
        
        total_resumes = len(request.resumes)
        logging.info(f"🤖 Processing {total_resumes} resumes for workflow {request.workflow_id}")
        logging.info(f"⚡ Using 5 parallel workers")
        logging.info(f"⏱️ Estimated time: {(total_resumes / 5) * 30:.0f}-{(total_resumes / 5) * 40:.0f} seconds")
        
        # Process resumes in parallel batches of 5
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Create futures for all resume processing tasks
            futures = [
                loop.run_in_executor(
                    executor,
                    process_single_resume_sequential,
                    resume["resume_id"],
                    resume["resume_text"],
                    request.jd_text,
                    app_graph,
                    parser,
                    idx + 1,
                    total_resumes,
                    request.workflow_id,
                    request.jd_id
                )
                for idx, resume in enumerate(request.resumes)
            ]
            
            # Wait for all resumes to complete (5 at a time)
            logging.info(f"🚀 Starting parallel processing (5 workers)...")
            results = await asyncio.gather(*futures)
            logging.info(f"✅ All {total_resumes} resumes processed!")
        
        total_time = time.time() - start_time
        
        response = {
            "workflow_id": request.workflow_id,
            "total_resumes": len(results),
            "processing_time_ms": int(total_time * 1000),
            "results": results
        }
        
        logging.info(f"✅ Batch processing complete for workflow {request.workflow_id}")
        logging.info(f"⏱️ Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logging.info(f"⚡ Average time per resume: {total_time/len(results):.1f} seconds")
        logging.info(f"🚀 Speedup: {5}x faster than sequential!")
        
        return response
        
    except Exception as e:
        logging.error(f"❌ Batch comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Single Resume Extraction Endpoint
# ============================================
class ResumeExtractionRequest(BaseModel):
    resume_text: str

@app.post("/extract-resume")
def extract_resume(request: ResumeExtractionRequest):
    """
    Extract structured data from a single resume
    Used during resume upload for one-time extraction
    
    Input: {"resume_text": "..."}
    Output: {"success": true, "extracted_data": {...}}
    """
    try:
        logging.info(f"🔍 Extracting data from resume ({len(request.resume_text)} chars)")
        
        parser = JsonOutputParser()
        graph = build_langgraph()
        app_graph = graph.compile()
        
        # Run extraction through LangGraph (only Resume Extractor runs)
        state = {"resume_text": request.resume_text}
        result_state = app_graph.invoke(state)
        
        # Parse extracted resume data
        resume_raw = result_state.get("resume_extracted", "")
        try:
            resume_data = json.loads(
                resume_raw.replace("```json", "").replace("```", "").strip()
            )
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse resume data: {e}")
            resume_data = {}
        
        logging.info(f"✅ Resume extraction complete")
        
        return {
            "success": True,
            "extracted_data": resume_data
        }
        
    except Exception as e:
        logging.error(f"❌ Resume extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Single JD Extraction Endpoint  
# ============================================
class JDExtractionRequest(BaseModel):
    jd_text: str

@app.post("/extract-jd")
def extract_jd(request: JDExtractionRequest):
    """
    Extract structured data from a single job description
    Used during JD creation for one-time extraction
    
    Input: {"jd_text": "..."}
    Output: {"success": true, "extracted_data": {...}}
    """
    try:
        logging.info(f"🔍 Extracting data from JD ({len(request.jd_text)} chars)")
        
        parser = JsonOutputParser()
        graph = build_langgraph()
        app_graph = graph.compile()
        
        # Run extraction through LangGraph (only JD Extractor runs)
        state = {"jd_text": request.jd_text}
        result_state = app_graph.invoke(state)
        
        # Parse extracted JD data
        jd_raw = result_state.get("jd_extracted", "")
        try:
            jd_data = json.loads(
                jd_raw.replace("```json", "").replace("```", "").strip()
            )
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse JD data: {e}")
            jd_data = {}
        
        logging.info(f"✅ JD extraction complete")
        
        return {
            "success": True,
            "extracted_data": jd_data
        }
        
    except Exception as e:
        logging.error(f"❌ JD extraction error: {e}", exc_info=True)
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
    uvicorn.run("backend_api:app", host="0.0.0.0", port=9000, reload=True)
