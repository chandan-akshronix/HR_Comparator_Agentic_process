# =============================================
# ✅ AI Resume–JD Matching Backend (MongoDB Atlas)
# =============================================

import sys, os, logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import main

# Fix import path for local testing
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment
load_dotenv()
logging.basicConfig(level=logging.INFO)

# MongoDB Connection (Atlas)
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hr_resume_comparator")

client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Initialize FastAPI
app = FastAPI(title="AI Resume–JD Matching Backend")

# Enable CORS for React Frontend
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class UploadJDRequest(BaseModel):
    jd_id: str
    designation: str
    text: str

class UploadResumeRequest(BaseModel):
    jd_id: str
    filename: str
    text: str

# Upload JD
@app.post("/upload/jd")
def upload_jd(req: UploadJDRequest):
    try:
        db["JobDescription"].insert_one({
            "_id": req.jd_id,
            "designation": req.designation,
            "text": req.text,
            "job_status": "pending"
        })
        logging.info(f"✅ JD uploaded: {req.jd_id}")
        return {"message": "JD uploaded successfully", "jd_id": req.jd_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload Resume
@app.post("/upload/resume")
def upload_resume(req: UploadResumeRequest):
    try:
        db["Resume"].insert_one({
            "jd_id": req.jd_id,
            "filename": req.filename,
            "text": req.text,
            "processed": False
        })
        logging.info(f"✅ Resume uploaded for JD {req.jd_id}")
        return {"message": "Resume uploaded successfully", "filename": req.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Trigger Matching Agent
@app.post("/run_agent/{jd_id}")
def run_agent(jd_id: str, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(main, jd_id)
        return {"message": f"Matching process started for JD {jd_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get Results
@app.get("/results/{jd_id}")
def get_results(jd_id: str):
    try:
        results = list(db["resume_result"].find({"jd_id": jd_id}, {"_id": 0}))
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "AI Resume–JD Matching Backend connected to MongoDB Atlas ✅"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)
