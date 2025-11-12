import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hr_resume_comparator")

def get_client():
    return MongoClient(MONGODB_URL)

def fetch_jd_by_id(jd_id: str):
    client = get_client()
    db = client[DATABASE_NAME]
    jd = db["JobDescription"].find_one({"_id": jd_id})
    client.close()
    return jd.get("text", "") if jd else ""

def fetch_resumes_by_jd_id(jd_id: str):
    client = get_client()
    db = client[DATABASE_NAME]
    resumes = []
    for doc in db["Resume"].find({"jd_id": jd_id, "processed": {"$ne": True}}):
        resumes.append({"_id": str(doc["_id"]), "text": doc.get("text", "")})
    client.close()
    return resumes
