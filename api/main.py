"""FastAPI service: REST API plus the web chat at /.

    uvicorn api.main:app --port 8000
    # chat UI: http://localhost:8000      OpenAPI docs: http://localhost:8000/docs

Environment variables:
    MODEL_BACKEND         "pytorch" (default), "onnx", "transformer", or "transformer-onnx"
    MODEL_PATH            model file or directory (defaults to the v1 BiLSTM files)
    CONFIDENCE_THRESHOLD  optional override; by default the threshold chosen on validation
                          (serving.json next to the model) is used
    INTENTS_FILE          intents file with the chat responses (default data/intents.json)
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    BatchPrediction,
    BatchPredictRequest,
    ChatRequest,
    ChatResponse,
    Health,
    IntentScore,
    ModelInfo,
    Prediction,
    PredictRequest,
)
from src.config import resolve
from src.dialogue import Dialogue, load_responses
from src.safety import red_flag
from src.serving import load_classifier  # torch-free; the chosen backend is imported lazily

DISCLAIMER = (
    "This service is a demo intent classifier and does not provide medical advice. "
    "Consult a healthcare professional; in an emergency, contact local emergency services."
)
UNSURE_INTENT = "unsure"
OUT_OF_SCOPE = "out_of_scope"  # explicit intent in dataset v2 models
UNSURE_MESSAGE = "I'm not sure what you mean. Could you rephrase, or describe your main symptom?"
TOP_K = 3
WEB_DIR = resolve("web")
PAGE_HEADERS = {
    # The page loads nothing from other origins, so lock it down.
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def create_app(model_path=None, confidence_threshold=None, backend=None, intents_file=None):
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
        app.state.dialogue = Dialogue(clf, app.state.confidence_threshold, load_responses(intents_file))
        yield
        app.state.classifier = None
        app.state.dialogue = None

    app = FastAPI(
        title="Medical Intent Classifier",
        version="0.4.0",
        description=f"Classifies a healthcare chatbot message into an intent, and runs a short symptom "
                    f"conversation.\n\n**{DISCLAIMER}**",
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
            confident = top[0].confidence >= threshold and top[0].intent != OUT_OF_SCOPE
            flag = red_flag(text)
            results.append(Prediction(
                text=text,
                intent=top[0].intent if confident or top[0].intent == OUT_OF_SCOPE else UNSURE_INTENT,
                confidence=top[0].confidence,
                is_confident=confident,
                top_k=top,
                message=flag[1] if flag else (None if confident else UNSURE_MESSAGE),
                safety_alert=flag[0] if flag else None,
                disclaimer=DISCLAIMER,
            ))
        return results

    @app.get("/", include_in_schema=False)
    def root():
        index = WEB_DIR / "index.html"
        if not index.exists():
            return RedirectResponse(url="/docs")
        return FileResponse(index, headers={**PAGE_HEADERS, "Cache-Control": "no-cache"})

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def old_ui():
        return RedirectResponse(url="/")  # the Gradio UI used to live here

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

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        """One conversation turn. Send back the returned `state` with the next message."""
        classifier(request)  # 503 while the model is loading
        return ChatResponse(**request.app.state.dialogue.respond(body.message, body.state), disclaimer=DISCLAIMER)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


app = create_app()
