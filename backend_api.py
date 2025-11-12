import sys, os, logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import main_workflow

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

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)
