import pytest
from fastapi.testclient import TestClient

pytest.importorskip("gradio")

from api.main import UNSURE_MESSAGE, create_app  # noqa: E402
from api.ui import make_responder  # noqa: E402
from src.predict import IntentClassifier  # noqa: E402
from tests.conftest import LABELS  # noqa: E402

RESPONSES = {label: [f"reply for {label}"] for label in LABELS}


def responder(tiny_checkpoint, threshold):
    clf = IntentClassifier(tiny_checkpoint)
    return make_responder(lambda: clf, lambda: threshold, UNSURE_MESSAGE, RESPONSES)


def test_confident_reply_uses_intent_response(tiny_checkpoint):
    box, history, scores = responder(tiny_checkpoint, 0.0)("I have a fever", [])
    assert box == ""
    assert [m["role"] for m in history] == ["user", "assistant"]
    top = max(scores, key=scores.get)
    assert history[1]["content"].startswith(f"reply for {top}")
    assert set(scores) == set(LABELS) and sum(scores.values()) == pytest.approx(1.0)


def test_low_confidence_reply_asks_to_rephrase(tiny_checkpoint):
    _, history, _ = responder(tiny_checkpoint, 1.01)("I have a fever", [{"role": "user", "content": "hi"}])
    assert len(history) == 3
    assert history[-1]["content"].startswith(UNSURE_MESSAGE)


def test_blank_message_changes_nothing(tiny_checkpoint):
    assert responder(tiny_checkpoint, 0.5)("   ", []) == ("", [], None)


def test_ui_is_mounted_and_root_redirects(tiny_checkpoint):
    with TestClient(create_app(model_path=tiny_checkpoint, enable_ui=True)) as c:
        assert c.get("/", follow_redirects=False).headers["location"] == "/ui"
        assert c.get("/ui/").status_code == 200
        assert c.post("/predict", json={"text": "hello"}).status_code == 200
