import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "pod_1")

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

def get_client():
    return MongoClient(MONGODB_URL, **get_connection_options())

# ---------------------------
# JD & Resume Fetchers
# ---------------------------
def fetch_jd_by_id(jd_id: str):
    """Fetch JD text from JobDescription collection."""
    client = get_client()
    db = client[DATABASE_NAME]
    jd = db["JobDescription"].find_one({"_id": jd_id})
    client.close()
    return jd.get("text", "") if jd else ""

def fetch_resume_by_id(resume_id):
    """Fetch resume text by ObjectId or string from Resume collection."""
    client = get_client()
    db = client[DATABASE_NAME]
    resume = db["Resume"].find_one({"_id": resume_id})
    client.close()
    return resume.get("text", "") if resume else ""

# ---------------------------
# Workflow Fetcher
# ---------------------------
def fetch_workflow_by_id(workflow_id: str):
    """
    Fetch a workflow execution record by workflow_id
    from DB: hr_resume_comparator.hr_resume_comparator.workflow_executions
    """
    client = get_client()
    db = client["hr_resume_comparator"]
    doc = db["hr_resume_comparator.workflow_executions"].find_one({"workflow_id": workflow_id})
    client.close()
    return doc
