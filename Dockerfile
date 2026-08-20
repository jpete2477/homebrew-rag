# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Install the CPU-only PyTorch build first: the default wheel pulls ~2 GB of
# CUDA libraries that are dead weight on a CPU inference host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /models /app/documents \
    && chown -R rag:rag /models /app
USER rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "homebrew_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
