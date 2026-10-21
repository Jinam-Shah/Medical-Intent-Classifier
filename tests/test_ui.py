import random

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("gradio")

from api.main import create_app  # noqa: E402
from api.ui import make_responder, symptom_list, unsure_reply  # noqa: E402
from src.predict import IntentClassifier  # noqa: E402
from src.safety import CRISIS_MESSAGE, EMERGENCY_MESSAGE  # noqa: E402
from tests.conftest import LABELS  # noqa: E402

RESPONSES = {label: [f"reply for {label}"] for label in LABELS}


def responder(tiny_checkpoint, threshold):
    clf = IntentClassifier(tiny_checkpoint)
    return make_responder(lambda: clf, lambda: threshold, RESPONSES, rng=random.Random(0)), clf


def test_confident_reply_is_the_intent_response_without_debug_text(tiny_checkpoint):
    reply, clf = responder(tiny_checkpoint, 0.0)
    top = clf.predict("I have a fever")["intent"]
    assert reply("I have a fever", []) == f"reply for {top}"


def test_low_confidence_reply_asks_to_rephrase(tiny_checkpoint):
    reply, _ = responder(tiny_checkpoint, 1.01)
    assert reply("I have a fever", []) == unsure_reply(RESPONSES)


@pytest.mark.parametrize("text, message", [
    ("I have chest pain", EMERGENCY_MESSAGE),
    ("I can't breathe", EMERGENCY_MESSAGE),
    ("i want to kill myself", CRISIS_MESSAGE),
])
def test_red_flags_override_the_model(tiny_checkpoint, text, message):
    reply, _ = responder(tiny_checkpoint, 0.0)  # threshold 0: the model would otherwise always answer
    assert message in reply(text, [])


def test_blank_message(tiny_checkpoint):
    reply, _ = responder(tiny_checkpoint, 0.5)
    assert reply("   ", []) == "Please type a message."


def test_ui_is_mounted_without_footer_links_and_root_redirects(tiny_checkpoint):
    with TestClient(create_app(model_path=tiny_checkpoint, enable_ui=True)) as c:
        assert c.get("/", follow_redirects=False).headers["location"] == "/ui"
        page = c.get("/ui/")
        assert page.status_code == 200
        assert '"footer_links":[]' in page.text.replace(" ", "")
        assert c.post("/predict", json={"text": "hello"}).status_code == 200


def test_symptom_list_comes_from_intents():
    responses = {"fever": ["x"], "nausea_vomiting": ["y"], "sore_throat": ["z"], "greeting": ["hi"],
                 "out_of_scope": ["no"]}
    assert symptom_list(responses) == "fever, nausea and sore throat"
