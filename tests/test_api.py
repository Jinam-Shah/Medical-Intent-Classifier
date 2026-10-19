import json

import pytest
from fastapi.testclient import TestClient

from api.main import UNSURE_INTENT, create_app
from api.schemas import MAX_BATCH_SIZE, MAX_TEXT_CHARS
from src.predict import SERVING_FILE
from tests.conftest import LABELS


@pytest.fixture
def client(tiny_checkpoint):
    # threshold 0 so every prediction counts as confident
    with TestClient(create_app(model_path=tiny_checkpoint, confidence_threshold=0.0)) as c:
        yield c


@pytest.fixture
def unsure_client(tiny_checkpoint):
    # a threshold above any possible confidence forces the fallback path
    with TestClient(create_app(model_path=tiny_checkpoint, confidence_threshold=1.01)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["labels"] == LABELS
    assert body["backend"] == "pytorch"
    assert body["num_parameters"] > 0
    assert "not provide medical advice" in body["disclaimer"]


def test_predict_returns_intent_confidence_and_top3(client):
    body = client.post("/predict", json={"text": "I have a fever"}).json()
    assert body["intent"] in LABELS
    assert body["is_confident"] is True
    assert body["message"] is None
    assert len(body["top_k"]) == 3
    assert body["top_k"][0]["intent"] == body["intent"]
    assert body["top_k"][0]["confidence"] == pytest.approx(body["confidence"])
    confidences = [s["confidence"] for s in body["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["disclaimer"]


def test_low_confidence_returns_unsure(unsure_client):
    body = unsure_client.post("/predict", json={"text": "I have a fever"}).json()
    assert body["intent"] == UNSURE_INTENT
    assert body["is_confident"] is False
    assert "rephrase" in body["message"]
    assert body["top_k"][0]["intent"] in LABELS  # the best guess is still reported


def test_batch_matches_single_predictions(client):
    texts = ["hello", "i have a cough", "something unseen"]
    batch = client.post("/predict/batch", json={"texts": texts}).json()["results"]
    assert [r["text"] for r in batch] == texts
    for text, result in zip(texts, batch, strict=True):
        single = client.post("/predict", json={"text": text}).json()
        assert single["intent"] == result["intent"]
        assert single["confidence"] == pytest.approx(result["confidence"], abs=1e-6)


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   "},
    {"text": "a" * (MAX_TEXT_CHARS + 1)},
    {},
    {"text": 123},
])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("texts", [[], ["ok"] * (MAX_BATCH_SIZE + 1), ["fine", ""]])
def test_batch_rejects_invalid_input(client, texts):
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict/batch" in client.get("/openapi.json").json()["paths"]


def test_threshold_defaults_to_serving_file(tiny_checkpoint, tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(tiny_checkpoint.read_bytes())
    (tmp_path / SERVING_FILE).write_text(json.dumps({"temperature": 2.0, "confidence_threshold": 0.66}))
    with TestClient(create_app(model_path=ckpt)) as c:
        info = c.get("/model-info").json()
    assert (info["confidence_threshold"], info["temperature"]) == (0.66, 2.0)
