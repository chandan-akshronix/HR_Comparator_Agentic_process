# Builder stage
FROM python:3.10-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN adduser --disabled-password --gecos '' appuser
WORKDIR /app
COPY --from=builder /venv /venv
COPY . .
RUN mkdir -p /app/output && chown appuser:appuser /app/output
USER appuser

# THIS LINE IS THE FIX
ENV PATH="/venv/bin:$PATH"

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=20s --start-period=120s --retries=15 \
    CMD wget --spider --quiet http://localhost:9000/ || exit 1

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "9000"]
