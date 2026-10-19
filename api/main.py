"""FastAPI service.

    uvicorn api.main:app --port 8000
    # OpenAPI docs at http://localhost:8000/docs

Environment variables:
    MODEL_PATH            checkpoint path (default outputs/models/bilstm/model.pt)
    CONFIDENCE_THRESHOLD  below this, the API answers "unsure" (default 0.75)
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    BatchPrediction,
    BatchPredictRequest,
    Health,
    IntentScore,
    ModelInfo,
    Prediction,
    PredictRequest,
)
from src.model import count_parameters
from src.predict import DEFAULT_CHECKPOINT, IntentClassifier

DISCLAIMER = (
    "This service is a demo intent classifier and does not provide medical advice. "
    "Consult a healthcare professional; in an emergency, contact local emergency services."
)
UNSURE_INTENT = "unsure"
UNSURE_MESSAGE = "I'm not sure what you mean. Could you rephrase, or describe your main symptom?"
DEFAULT_THRESHOLD = 0.75
TOP_K = 3


def create_app(model_path=None, confidence_threshold=None):
    model_path = model_path or os.getenv("MODEL_PATH", DEFAULT_CHECKPOINT)
    if confidence_threshold is None:
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", DEFAULT_THRESHOLD))

    @asynccontextmanager
    async def lifespan(app):
        app.state.classifier = IntentClassifier(model_path)
        yield
        app.state.classifier = None

    app = FastAPI(
        title="Medical Intent Classifier",
        version="0.3.0",
        description=f"Classifies a healthcare chatbot message into one of 12 intents.\n\n**{DISCLAIMER}**",
        lifespan=lifespan,
    )
    app.state.confidence_threshold = confidence_threshold

    def classifier(request: Request) -> IntentClassifier:
        clf = getattr(request.app.state, "classifier", None)
        if clf is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return clf

    def to_predictions(clf, texts):
        probs = clf.predict_proba(texts)
        results = []
        for text, p in zip(texts, probs, strict=True):
            order = np.argsort(p)[::-1][:TOP_K]
            top = [IntentScore(intent=clf.labels[i], confidence=float(p[i])) for i in order]
            confident = top[0].confidence >= confidence_threshold
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

    @app.get("/health", response_model=Health)
    def health(request: Request):
        loaded = getattr(request.app.state, "classifier", None) is not None
        return Health(status="ok" if loaded else "loading", model_loaded=loaded)

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request):
        clf = classifier(request)
        return ModelInfo(
            model_name=clf.metadata.get("model_name", "bilstm_attention"),
            backend="pytorch",
            labels=clf.labels,
            num_parameters=count_parameters(clf.model),
            confidence_threshold=confidence_threshold,
            best_epoch=clf.metadata.get("best_epoch"),
            trained_at=clf.metadata.get("trained_at"),
            disclaimer=DISCLAIMER,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(body: PredictRequest, request: Request):
        return to_predictions(classifier(request), [body.text])[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(body: BatchPredictRequest, request: Request):
        return BatchPrediction(results=to_predictions(classifier(request), body.texts))

    return app


app = create_app()
