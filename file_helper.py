import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def store_to_file(result: dict, resume_id: str):
    """
    Stores each resume's result to a JSON file.
    """
    try:
        output_dir = os.getenv("OUTPUT_DIR", "output")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{resume_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logging.info(f"✅ Saved result to {file_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save result for {resume_id}: {e}")
