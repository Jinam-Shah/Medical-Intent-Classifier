# syntax=docker/dockerfile:1
#
# Packages the already-trained, already-evaluated model; it does not train.
# (Training on Linux takes a different numerical path than the run recorded in METRICS.md,
#  so an image that trained itself would serve a model nobody evaluated. See DECISIONS D18.)
#
# Model source (build arg MODEL_SOURCE):
#   local (default)  copy outputs/models/bilstm/ from the build context; create it first with
#                    python -m src.train && python -m src.threshold --split val && python -m src.export_onnx
#   hub              download a pinned revision from the Hugging Face Hub
#                    (HF_MODEL_REPO, HF_MODEL_REVISION); used by Render and CI
#
# Targets:
#   serve          default: ONNX Runtime API + Gradio chat UI at /ui  (what gets deployed)
#   serve-api      ONNX Runtime API only, no UI
#   serve-pytorch  PyTorch-backed API (kept for the size comparison; local model only)
#
#   docker build -t medical-intent-classifier .
#   docker build --build-arg MODEL_SOURCE=hub --build-arg HF_MODEL_REPO=user/repo \
#                --build-arg HF_MODEL_REVISION=<commit> -t medical-intent-classifier .
#   docker run --rm -p 8000:8000 medical-intent-classifier      # http://localhost:8000

ARG PYTHON_IMAGE=python:3.11.9-slim-bookworm
ARG MODEL_SOURCE=local

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

# ---------------------------------------------------------------- model artifacts
FROM scratch AS model-local
COPY outputs/models/bilstm/model.onnx \
     outputs/models/bilstm/model_meta.json \
     outputs/models/bilstm/serving.json \
     /model/

FROM base AS model-hub
ARG HF_MODEL_REPO
ARG HF_MODEL_REVISION
COPY src/__init__.py src/config.py src/serving.py src/hub.py src/
RUN test -n "$HF_MODEL_REPO" && test -n "$HF_MODEL_REVISION" \
 && pip install huggingface_hub==2.2.0 PyYAML==6.0.3 numpy==2.4.6 \
 && python -m src.hub download --repo "$HF_MODEL_REPO" --revision "$HF_MODEL_REVISION" --dest /model

FROM model-${MODEL_SOURCE} AS model

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

# ---------------------------------------------------------------- serve-api: ONNX Runtime, no torch
FROM base AS serve-api
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt \
 && python -m nltk.downloader -d "$NLTK_DATA" punkt_tab
COPY src src
COPY api api
COPY --from=model /model/ /app/model/
USER app
# one ONNX Runtime thread, no spin-waiting: free hosting tiers give a fraction of one CPU
ENV MODEL_BACKEND=onnx MODEL_PATH=/app/model/model.onnx ORT_NUM_THREADS=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

# ---------------------------------------------------------------- serve (default, last stage): + Gradio UI
FROM serve-api AS serve
USER root
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY data/intents.json data/intents.json
USER app
ENV ENABLE_UI=1 GRADIO_ANALYTICS_ENABLED=False
