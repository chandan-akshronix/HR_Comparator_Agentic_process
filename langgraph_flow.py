import logging
from langgraph.graph import StateGraph
from azure_llm import get_azure_llm

# ==========================================
# Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = get_azure_llm()
logging.info("✅ Azure OpenAI LLM initialized for LangGraph pipeline.")

# ==========================================
# PROMPTS
# ==========================================

JD_PROMPT = """You are a senior HR specialist. Extract **structured, factual** details from this Job Description.

Job Description:
{jd_text}

Return clean JSON:
{{
  "Position": "",
  "Experience_Required_Years": "",
  "Must_Have_Skills": [],
  "Nice_To_Have_Skills": [],
  "Education": "",
  "Responsibilities": [],
  "Soft_Skills": [],
  "Location": "",
  "Industry": ""
}}
"""

RESUME_PROMPT = """You are a professional recruiter. Parse and extract **key structured details** from this resume.

Resume Text:
{resume_text}

Return clean JSON:
{{
  "Name": "",
  "Email": "",
  "Mobile": "",
  "Total_Experience_Years": "",
  "Technical_Skills": [],
  "Soft_Skills": [],
  "Education": "",
  "Projects": [],
  "Certifications": [],
  "Domain_Experience": "",
  "Current_Location": "",
  "Career_History": [
      {{
        "Company": "",
        "Job_Title": "",
        "Start_Date": "",
        "End_Date": ""
      }}
  ]
}}

**IMPORTANT:** 
- Career_History MUST be ordered chronologically with the **MOST RECENT position FIRST** (reverse chronological order).
- The first entry in Career_History should be the candidate's current/latest job.
- For current positions, use "Present" or "Current" for End_Date if not specified.
"""

COMPARATOR_PROMPT = """You are an experienced recruiter evaluating a candidate for hiring.

Compare the Job Description and Resume below with **human-level HR reasoning**.

---
Job Description:
{jd_extracted}

Resume:
{resume_extracted}
---

Evaluate using 3 recruiter mind layers:

### 1. FIT & COMPETENCE (Can they do the job?)
Calculate individual scores (0-100) for each parameter, then compute total_score using the weighted formula below.

- Technical Skills Score (0-100) - Match between required and candidate skills
- Experience Relevance Score (0-100) - How well experience aligns with requirements
- Project/Domain Alignment Score (0-100) - Relevance of projects and domain expertise
- Education & Certifications Score (0-100) - Match of education and certifications
- Soft Skills Score (0-100) - Alignment of soft skills with job requirements
- Location / Availability Score (0-100) - Location match and availability
- Stability Score (0-100) - **IMPORTANT:** If candidate has NO professional experience (no Career_History entries or empty Career_History), Stability Score MUST be 0. For candidates with experience: Lower score for frequent job changes, higher for stable career (longer tenure, fewer gaps).
- Overqualified Score (0-100) - Calculate based on: i) extra relevant experience beyond requirements, or ii) equal/more relevant experience with highly valuable additional skills. If candidate doesn't meet either criteria, score should be 0.

**Total Score Formula (calculate after individual scores):**
total_score = (Technical_Skills_Score × 0.30) + (Experience_Relevance_Score × 0.20) + (Project_Domain_Alignment_Score × 0.15) + (Education_Certifications_Score × 0.10) + (Soft_Skills_Score × 0.05) + (Location_Availability_Score × 0.05) + (Stability_Score × 0.05) + (Overqualified_Score × 0.10)

Note: Apply risk penalties and growth bonuses AFTER calculating base total_score.

### 2. RISK & RELIABILITY (Should I trust this hire?)
Look for potential risks:
- Frequent job changes
- Skill exaggeration
- Career inconsistency
- Overqualification or underqualification
- Culture or communication mismatch
- Availability or relocation risk

Each risk reduces 2–10 points.

### 3. GROWTH & VALUE (Will this person grow?)
Add bonus points (+2 to +10) for:
- Continuous learning
- Leadership / mentoring
- Cross-domain knowledge
- Problem-solving & adaptability
- Cultural alignment or passion

---

**Return structured JSON only**:

{{
  "fit_category": "Best Fit / Partial Fit / Not Fit",
  "total_score": <calculated number 0-100 using the weighted formula above>,
  "parameter_breakdown": {{
    "Skill_Score": "",
    "Experience_Score": "",
    "Project_Score": "",
    "Education_Score": "",
    "Soft_Skill_Score": "",
    "Location_Score": "",
    "Stability_Score": "",
    "Overqualified_Score": ""
  }},
  "risk_factors": ["..."],
  "growth_signals": ["..."],
  "recruiter_confidence": "High / Medium / Low",
  "selection_reason": "Detailed recruiter-style explanation combining skill match, risk, and growth reasoning."
}}
Be consistent, balanced, and think like a human recruiter minimizing hiring risk.
"""

# ==========================================
# LLM Nodes
# ==========================================

def jd_extractor_node(state: dict) -> dict:
    """Extracts structured JD info using LLM."""
    # Skip if already extracted (for optimization)
    if state.get("jd_extracted"):
        logging.info("⏭️ JD already extracted, skipping...")
        return state
    
    logging.info("🧩 JD Extractor node running...")
    jd_text = state.get("jd_text", "")

    if not jd_text:
        logging.warning("⚠️ JD text missing in state.")
        state["jd_extracted"] = "{}"
        return state

    try:
        response = llm.invoke(JD_PROMPT.format(jd_text=jd_text))
        state["jd_extracted"] = response.content
        logging.info("✅ JD extraction completed.")
    except Exception as e:
        logging.error(f"❌ JD extractor error: {e}")
        state["jd_extracted"] = "{}"
    return state


def resume_extractor_node(state: dict) -> dict:
    """Extracts structured Resume info using LLM."""
    # Skip if already extracted (for optimization)
    if state.get("resume_extracted"):
        logging.info("⏭️ Resume already extracted, skipping...")
        return state
        
    logging.info("🧾 Resume Extractor node running...")
    resume_text = state.get("resume_text", "")

    if not resume_text:
        logging.warning("⚠️ Resume text missing in state.")
        state["resume_extracted"] = "{}"
        return state

    try:
        response = llm.invoke(RESUME_PROMPT.format(resume_text=resume_text))
        state["resume_extracted"] = response.content
        logging.info("✅ Resume extraction completed.")
    except Exception as e:
        logging.error(f"❌ Resume extractor error: {e}")
        state["resume_extracted"] = "{}"
    return state


def comparator_node(state: dict) -> dict:
    """Compares JD and Resume using LLM with recruiter-style reasoning."""
    logging.info("⚖️ Comparator node running (human recruiter logic)...")

    jd_extracted = state.get("jd_extracted", "")
    resume_extracted = state.get("resume_extracted", "")

    if not jd_extracted or not resume_extracted:
        logging.warning("⚠️ Missing extracted data for comparison.")
        state["comparison_result"] = "{}"
        return state

    try:
        response = llm.invoke(COMPARATOR_PROMPT.format(
            jd_extracted=jd_extracted,
            resume_extracted=resume_extracted
        ))
        state["comparison_result"] = response.content
        logging.info("✅ Comparison completed.")
    except Exception as e:
        logging.error(f"❌ Comparator node error: {e}")
        state["comparison_result"] = "{}"
    return state

# ==========================================
# LangGraph Builder
# ==========================================

def build_langgraph():
    """
    Builds a flexible recruiter workflow graph with 3 agents:
    JD_Extractor → Resume_Extractor → Comparator
    
    Each node auto-skips if data already exists, enabling:
    - Full pipeline: Pass jd_text + resume_text (runs all 3)
    - Optimized pipeline: Pass jd_extracted + resume_extracted (skips to Comparator)
    - JD extraction only: Pass jd_text only (runs JD_Extractor, skips rest)
    - Resume extraction only: Pass resume_text only (runs Resume_Extractor, skips rest)
    """
    g = StateGraph(dict)

    # Add all 3 agent nodes
    g.add_node("JD_Extractor", jd_extractor_node)
    g.add_node("Resume_Extractor", resume_extractor_node)
    g.add_node("Comparator", comparator_node)

    # Sequential edges (nodes auto-skip if data exists)
    g.add_edge("JD_Extractor", "Resume_Extractor")
    g.add_edge("Resume_Extractor", "Comparator")

    g.set_entry_point("JD_Extractor")
    g.set_finish_point("Comparator")

    logging.info("✅ LangGraph flexible pipeline ready (JD → Resume → Comparator with auto-skip).")
    return g
