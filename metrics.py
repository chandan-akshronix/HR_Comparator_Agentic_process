# metrics.py - Custom Prometheus Metrics for AI Agent
"""
Custom business metrics for monitoring the AI Resume Matching Agent.
These metrics track AI/LLM operations and matching performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from contextlib import contextmanager

# ============================================
# Application Info
# ============================================
app_info = Info(
    'hr_ai_agent_app',
    'HR AI Agent application information'
)
app_info.info({
    'version': '1.0.0',
    'service': 'hr-ai-agent',
    'environment': 'production',
    'llm_provider': 'azure-openai'
})

# ============================================
# LLM/AI Metrics
# ============================================
llm_requests_total = Counter(
    'hr_llm_requests_total',
    'Total LLM API requests',
    ['model', 'operation', 'status']  # operation: jd_extract/resume_extract/compare
)

llm_request_duration = Histogram(
    'hr_llm_request_duration_seconds',
    'LLM API request duration',
    ['model', 'operation'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

llm_tokens_used = Counter(
    'hr_llm_tokens_used_total',
    'Total tokens consumed by LLM',
    ['model', 'type']  # type: prompt/completion
)

llm_rate_limits = Counter(
    'hr_llm_rate_limits_total',
    'Total LLM rate limit errors',
    ['model']
)

# ============================================
# Agent Pipeline Metrics
# ============================================
agent_extractions_total = Counter(
    'hr_agent_extractions_total',
    'Total extraction operations by agent type',
    ['agent_type', 'status']  # agent_type: jd_extractor/resume_extractor/comparator
)

agent_extraction_duration = Histogram(
    'hr_agent_extraction_duration_seconds',
    'Time spent on extraction by agent',
    ['agent_type'],
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

pipeline_runs_total = Counter(
    'hr_pipeline_runs_total',
    'Total 3-agent pipeline runs',
    ['status']  # success/failed/partial
)

pipeline_duration = Histogram(
    'hr_pipeline_duration_seconds',
    'Total 3-agent pipeline duration per resume',
    buckets=[5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]
)

# ============================================
# Batch Processing Metrics
# ============================================
batch_jobs_total = Counter(
    'hr_batch_jobs_total',
    'Total batch comparison jobs',
    ['status']
)

batch_size = Histogram(
    'hr_batch_size',
    'Distribution of batch sizes (resumes per batch)',
    buckets=[1, 5, 10, 15, 20, 25, 30, 50, 100]
)

batch_duration = Histogram(
    'hr_batch_duration_seconds',
    'Total batch processing duration',
    buckets=[30, 60, 120, 180, 300, 600, 900, 1800]
)

batch_throughput = Gauge(
    'hr_batch_throughput_resumes_per_minute',
    'Current batch processing throughput'
)

parallel_workers_active = Gauge(
    'hr_parallel_workers_active',
    'Number of parallel workers currently active'
)

# ============================================
# Match Score Metrics  
# ============================================
match_scores_generated = Histogram(
    'hr_ai_match_scores_generated',
    'Distribution of AI-generated match scores',
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

match_categories = Counter(
    'hr_ai_match_categories_total',
    'Count of matches by fit category',
    ['category']  # Best Fit, Partial Fit, Not Fit, Error
)

stability_scores = Histogram(
    'hr_stability_scores',
    'Distribution of candidate stability scores',
    buckets=[0, 20, 40, 60, 80, 100]
)

# ============================================
# Error Metrics
# ============================================
parsing_errors_total = Counter(
    'hr_ai_parsing_errors_total',
    'Total JSON parsing errors from LLM responses',
    ['operation']  # jd_parse/resume_parse/comparison_parse
)

processing_errors_total = Counter(
    'hr_ai_processing_errors_total',
    'Total resume processing errors',
    ['error_type']  # timeout/llm_error/validation_error/unknown
)

# ============================================
# Queue/Backlog Metrics
# ============================================
resumes_queued = Gauge(
    'hr_resumes_queued',
    'Number of resumes waiting to be processed'
)

resumes_processing = Gauge(
    'hr_resumes_processing',
    'Number of resumes currently being processed'
)

# ============================================
# Helper Functions
# ============================================
@contextmanager
def track_llm_request(model: str, operation: str):
    """Context manager to track LLM request duration and status"""
    start = time.time()
    status = 'success'
    try:
        yield
    except Exception as e:
        status = 'failed'
        if 'rate limit' in str(e).lower():
            llm_rate_limits.labels(model=model).inc()
        raise
    finally:
        duration = time.time() - start
        llm_requests_total.labels(model=model, operation=operation, status=status).inc()
        llm_request_duration.labels(model=model, operation=operation).observe(duration)

@contextmanager
def track_agent_extraction(agent_type: str):
    """Context manager to track agent extraction duration"""
    start = time.time()
    status = 'success'
    try:
        yield
    except Exception:
        status = 'failed'
        raise
    finally:
        duration = time.time() - start
        agent_extractions_total.labels(agent_type=agent_type, status=status).inc()
        agent_extraction_duration.labels(agent_type=agent_type).observe(duration)

@contextmanager
def track_pipeline_run():
    """Context manager to track full pipeline run"""
    start = time.time()
    status = 'success'
    try:
        yield
    except Exception:
        status = 'failed'
        raise
    finally:
        duration = time.time() - start
        pipeline_runs_total.labels(status=status).inc()
        pipeline_duration.observe(duration)

def track_batch_job(resume_count: int, duration: float, success: bool):
    """Track batch job metrics"""
    status = 'success' if success else 'failed'
    batch_jobs_total.labels(status=status).inc()
    batch_size.observe(resume_count)
    batch_duration.observe(duration)
    if duration > 0:
        throughput = (resume_count / duration) * 60
        batch_throughput.set(throughput)

def track_match_result(score: float, category: str, stability: float):
    """Track match result metrics"""
    match_scores_generated.observe(score)
    match_categories.labels(category=category).inc()
    stability_scores.observe(stability)

def track_parsing_error(operation: str):
    """Track JSON parsing error"""
    parsing_errors_total.labels(operation=operation).inc()

def track_processing_error(error_type: str):
    """Track processing error"""
    processing_errors_total.labels(error_type=error_type).inc()

def update_queue_status(queued: int, processing: int):
    """Update queue status gauges"""
    resumes_queued.set(queued)
    resumes_processing.set(processing)
