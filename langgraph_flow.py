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
# Note: JD_PROMPT and RESUME_PROMPT removed (extraction done during upload)
# Only Comparator prompt needed for matching

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
Use weighted scoring:
- Technical Skills (35%)
- Experience Relevance (25%)
- Project/Domain Alignment (15%)
- Education & Certifications (10%)
- Soft Skills (10%)
- Location / Availability (5%)

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
  "total_score": 0-100,
  "parameter_breakdown": {{
    "Skill_Score": "",
    "Experience_Score": "",
    "Project_Score": "",
    "Education_Score": "",
    "Soft_Skill_Score": "",
    "Location_Score": ""
  }},
  "risk_factors": ["..."],
  "growth_signals": ["..."],
  "recruiter_confidence": "High / Medium / Low",
  "selection_reason": "Detailed recruiter-style explanation combining skill match, risk, and growth reasoning."
}}
Be consistent, balanced, and think like a human recruiter minimizing hiring risk.
"""

# ==========================================
# LLM Node - Comparator Only
# ==========================================
# Note: jd_extractor_node and resume_extractor_node REMOVED
# Extraction now happens during upload and is stored in MongoDB

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
# LangGraph Builder - OPTIMIZED (Comparator Only)
# ==========================================

def build_langgraph():
    """
    OPTIMIZED: Only runs Comparator agent
    
    Uses pre-extracted JD and Resume data from MongoDB.
    - No JD_Extractor agent (data fetched from JobDescription collection)
    - No Resume_Extractor agent (data fetched from resume collection)
    - Only Comparator agent (1 Azure OpenAI call)
    
    Benefits:
    - 67% cost savings (1 API call instead of 3)
    - 2-3x faster processing
    - More reliable (uses validated extracted data)
    
    Input: Pre-extracted jd_extracted and resume_extracted from MongoDB
    Output: Comparison result with match_score and fit_category
    """
    g = StateGraph(dict)
    
    # Only add Comparator node (extraction agents removed)
    g.add_node("Comparator", comparator_node)
    
    # Direct entry and exit (no extraction steps)
    g.set_entry_point("Comparator")
    g.set_finish_point("Comparator")
    
    logging.info("✅ LangGraph OPTIMIZED pipeline ready (Comparator only - 67% cost savings).")
    return g
