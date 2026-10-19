# syntax=docker/dockerfile:1
#
# Packages the already-trained, already-evaluated model; it does not train.
# (Training on Linux takes a different numerical path than the run recorded in METRICS.md,
#  so an image that trained itself would serve a model nobody evaluated. See DECISIONS D18.)
#
# Create the artifacts first:
#   python -m src.train && python -m src.threshold --split val && python -m src.export_onnx
#
# Targets:
#   serve          default: slim API image with ONNX Runtime, no torch
#   serve-pytorch  PyTorch-backed API image (kept for the size comparison)
#
#   docker build -t medical-intent-classifier .
#   docker run --rm -p 8000:8000 medical-intent-classifier
#   # http://localhost:8000/docs

ARG PYTHON_IMAGE=python:3.11.9-slim-bookworm

# ---------------------------------------------------------------- shared base
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/share/nltk_data \
    PORT=8000
WORKDIR /app
RUN useradd --create-home --uid 10001 app

# ---------------------------------------------------------------- serve-pytorch (optional)
FROM base AS serve-pytorch
COPY requirements.txt .
# CPU-only torch first, so pip never pulls the multi-GB CUDA build
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
 && python -m nltk.downloader -d "$NLTK_DATA" punkt_tab
COPY src src
COPY api api
COPY outputs/models/bilstm/model.pt outputs/models/bilstm/serving.json /app/model/
USER app
ENV MODEL_BACKEND=pytorch MODEL_PATH=/app/model/model.pt
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

# ---------------------------------------------------------------- serve (default, last stage)
FROM base AS serve
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt \
 && python -m nltk.downloader -d "$NLTK_DATA" punkt_tab
COPY src src
COPY api api
COPY outputs/models/bilstm/model.onnx \
     outputs/models/bilstm/model_meta.json \
     outputs/models/bilstm/serving.json \
     /app/model/
USER app
ENV MODEL_BACKEND=onnx MODEL_PATH=/app/model/model.onnx
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
