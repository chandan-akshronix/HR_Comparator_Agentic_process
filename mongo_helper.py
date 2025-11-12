import json
from datetime import datetime
import logging
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hr_resume_comparator")

def store_to_mongo(result, resume_id):
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db["resume_result"]
    collection.update_one({"resume_id": resume_id}, {"$set": result}, upsert=True)
    client.close()


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_first_json_from_string(text: str):
    """
    Extracts the first JSON object from a string safely.
    """
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError:
        logging.warning("⚠️ Failed to decode JSON, returning raw text snippet.")
        return {"raw_output": text[:300]}

def store_to_mongo(result, resume_id: str, collection):
    """
    Stores comparison result to MongoDB.
    """
    if isinstance(result, str):
        result = extract_first_json_from_string(result)
    if not isinstance(result, dict):
        logging.error("❌ Result not a dictionary. Skipping storage.")
        return

    result.update({
        "resume_id": resume_id,
        "timestamp": datetime.utcnow().isoformat()
    })

    try:
        collection.update_one(
            {"resume_id": resume_id},
            {"$set": result},
            upsert=True
        )
        logging.info(f"✅ Stored resume {resume_id} in MongoDB.")
    except Exception as e:
        logging.error(f"❌ MongoDB error storing {resume_id}: {e}")
