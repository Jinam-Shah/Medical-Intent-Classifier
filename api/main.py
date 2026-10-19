"""FastAPI service.

    uvicorn api.main:app --port 8000
    # OpenAPI docs at http://localhost:8000/docs

Environment variables:
    MODEL_BACKEND         "pytorch" (default) or "onnx" (ONNX Runtime, no torch needed)
    MODEL_PATH            model file (default outputs/models/bilstm/model.pt, or model.onnx for onnx)
    CONFIDENCE_THRESHOLD  optional override; by default the threshold chosen on validation
                          (serving.json next to the checkpoint) is used
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from api.schemas import (
    BatchPrediction,
    BatchPredictRequest,
    Health,
    IntentScore,
    ModelInfo,
    Prediction,
    PredictRequest,
)
from src.serving import load_classifier  # torch-free; the chosen backend is imported lazily

DISCLAIMER = (
    "This service is a demo intent classifier and does not provide medical advice. "
    "Consult a healthcare professional; in an emergency, contact local emergency services."
)
UNSURE_INTENT = "unsure"
UNSURE_MESSAGE = "I'm not sure what you mean. Could you rephrase, or describe your main symptom?"
TOP_K = 3


def create_app(model_path=None, confidence_threshold=None, backend=None):
    backend = backend or os.getenv("MODEL_BACKEND", "pytorch")
    model_path = model_path or os.getenv("MODEL_PATH") or None
    if confidence_threshold is None and os.getenv("CONFIDENCE_THRESHOLD"):
        confidence_threshold = float(os.environ["CONFIDENCE_THRESHOLD"])

    @asynccontextmanager
    async def lifespan(app):
        clf = load_classifier(backend, model_path)
        app.state.classifier = clf
        app.state.confidence_threshold = (
            clf.confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        yield
        app.state.classifier = None

    app = FastAPI(
        title="Medical Intent Classifier",
        version="0.3.0",
        description=f"Classifies a healthcare chatbot message into one of 12 intents.\n\n**{DISCLAIMER}**",
        lifespan=lifespan,
    )

    def classifier(request: Request):
        clf = getattr(request.app.state, "classifier", None)
        if clf is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return clf

    def to_predictions(request, texts):
        clf, threshold = classifier(request), request.app.state.confidence_threshold
        probs = clf.predict_proba(texts)
        results = []
        for text, p in zip(texts, probs, strict=True):
            order = np.argsort(p)[::-1][:TOP_K]
            top = [IntentScore(intent=clf.labels[i], confidence=float(p[i])) for i in order]
            confident = top[0].confidence >= threshold
            results.append(Prediction(
                text=text,
                intent=top[0].intent if confident else UNSURE_INTENT,
                confidence=top[0].confidence,
                is_confident=confident,
                top_k=top,
                message=None if confident else UNSURE_MESSAGE,
                disclaimer=DISCLAIMER,
            ))
        return results

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=Health)
    def health(request: Request):
        loaded = getattr(request.app.state, "classifier", None) is not None
        return Health(status="ok" if loaded else "loading", model_loaded=loaded)

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request):
        clf = classifier(request)
        return ModelInfo(
            model_name=clf.metadata.get("model_name", "bilstm_attention"),
            backend=clf.backend,
            labels=clf.labels,
            num_parameters=clf.num_parameters,
            confidence_threshold=request.app.state.confidence_threshold,
            temperature=clf.temperature,
            best_epoch=clf.metadata.get("best_epoch"),
            trained_at=clf.metadata.get("trained_at"),
            disclaimer=DISCLAIMER,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(body: PredictRequest, request: Request):
        return to_predictions(request, [body.text])[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(body: BatchPredictRequest, request: Request):
        return BatchPrediction(results=to_predictions(request, body.texts))

    return app


app = create_app()
