import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.dialogue import (
    AFTER_ADVICE,
    DURATION_CHOICES,
    SEVERITY_CHOICES,
    Dialogue,
    clean_state,
    parse_duration,
    parse_severity,
    tailored_note,
)
from src.safety import EMERGENCY_MESSAGE

LABELS = ["capabilities", "cough", "dizziness", "ear_pain", "fever", "greeting", "out_of_scope", "stomach_ache",
          "thanks"]
KEYWORDS = {"stomach": "stomach_ache", "tummy": "stomach_ache", "fever": "fever", "cough": "cough",
            "ear": "ear_pain", "dizzy": "dizziness", "hello": "greeting", "hi": "greeting",
            "what can you": "capabilities", "thank": "thanks"}
RESPONSES = {label: [f"advice for {label}", f"more advice for {label}"] for label in LABELS}


class KeywordClassifier:
    """Deterministic stand-in for the model: the first matching keyword wins at 0.95."""
    labels = LABELS

    def predict_proba(self, texts):
        out = []
        for text in texts:
            p = np.full(len(LABELS), 0.05 / (len(LABELS) - 1))
            label = next((v for k, v in KEYWORDS.items() if k in text.lower()), "out_of_scope")
            p[LABELS.index(label)] = 0.95
            out.append(p)
        return np.array(out)


@pytest.fixture
def bot():
    return Dialogue(KeywordClassifier(), 0.5, RESPONSES)


def run(bot, *messages):
    state, replies = None, []
    for m in messages:
        r = bot.respond(m, state)
        state = r["state"]
        replies.append(r)
    return replies


@pytest.mark.parametrize("text, expected", [
    ("Started today", "under_1_day"), ("since this morning", "under_1_day"), ("1–3 days", "1_3_days"),
    ("since yesterday", "1_3_days"), ("a couple of days", "1_3_days"), ("4–7 days", "4_7_days"),
    ("about five days", "4_7_days"), ("Over a week", "over_week"), ("for two weeks", "over_week"),
    ("months", "over_week"), ("no idea", None),
])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Mild", "mild"), ("not too bad", "mild"), ("a bit", "mild"), ("Moderate", "moderate"),
    ("it's bad", "moderate"), ("Severe", "severe"), ("pretty bad", "severe"), ("it's killing me", "severe"),
    ("9/10", "severe"), ("5 out of 10", "moderate"), ("2/10", "mild"), ("hmm", None),
])
def test_parse_severity(text, expected):
    assert parse_severity(text) == expected


def test_full_symptom_flow(bot):
    first, second, third = run(bot, "my stomach hurts", "since yesterday", "pretty bad")
    assert first["intents"] == ["stomach_ache"] and first["quick_replies"][:4] == DURATION_CHOICES
    assert first["state"]["stage"] == "ask_duration"
    assert second["quick_replies"][:3] == SEVERITY_CHOICES and second["state"]["duration"] == "1_3_days"
    assert "advice for stomach_ache" in third["reply"] and "see a doctor soon" in third["reply"]
    assert third["state"]["stage"] == "idle" and third["quick_replies"] == AFTER_ADVICE


def test_message_with_duration_and_severity_skips_the_questions(bot):
    (reply,) = run(bot, "my tummy has been killing me for two weeks")
    assert "advice for stomach_ache" in reply["reply"] and reply["state"]["stage"] == "idle"


def test_multiple_symptoms_in_one_message(bot):
    first, _, last = run(bot, "I have a fever and a cough", "4-7 days", "mild")
    assert first["intents"] == ["fever", "cough"] and "them" in first["reply"]
    assert "advice for fever" in last["reply"] and "advice for cough" in last["reply"]


def test_new_symptom_during_follow_up_is_added(bot):
    _, second = run(bot, "my ear hurts", "actually i also feel dizzy")
    assert second["intents"] == ["ear_pain", "dizziness"] and second["state"]["stage"] == "ask_duration"


def test_skip_gives_advice_immediately(bot):
    _, second = run(bot, "i have a fever", "Skip")
    assert "advice for fever" in second["reply"] and second["state"]["stage"] == "idle"


def test_red_flag_overrides_everything_mid_conversation(bot):
    _, second = run(bot, "i have a fever", "now I have chest pain")
    assert second["kind"] == "emergency" and second["reply"] == EMERGENCY_MESSAGE
    assert second["state"]["stage"] == "idle" and second["state"]["symptoms"] == []


def test_conversational_and_out_of_scope(bot):
    hello, info, other, thanks = run(bot, "hello", "what can you do", "what's the weather", "thanks")
    assert hello["reply"].startswith("advice for greeting") or "greeting" in hello["reply"]
    assert info["intents"] == ["capabilities"]
    assert other["intents"] == ["out_of_scope"] and "I'm not sure I can help" in other["reply"]
    assert thanks["intents"] == ["thanks"]


def test_tampered_state_is_sanitised():
    state = clean_state({"stage": "ask_severity", "symptoms": ["fever", "<script>", 5], "duration": "forever",
                         "turns": -3, "extra": "x" * 100})
    assert state == {"stage": "ask_severity", "symptoms": ["fever"], "duration": None, "severity": None, "turns": 0}
    assert clean_state({"stage": "ask_duration", "symptoms": []})["stage"] == "idle"
    assert clean_state("not a dict")["stage"] == "idle"


def test_tailored_note():
    assert "see a doctor soon" in tailored_note(["fever"], "over_week", "mild")
    assert "emergency" in tailored_note(["shortness_of_breath"], None, "severe")
    assert "on their own" in tailored_note(["fever"], "under_1_day", "mild")


def test_chat_endpoint_and_web_page(tiny_checkpoint, tmp_path):
    import json
    intents = tmp_path / "intents.json"
    intents.write_text(json.dumps({"intents": [{"tag": t, "responses": [f"r {t}"]}
                                              for t in ["cough", "fever", "greeting"]]}))
    with TestClient(create_app(model_path=tiny_checkpoint, confidence_threshold=0.0,
                               intents_file=str(intents))) as c:
        page = c.get("/")
        assert page.status_code == 200 and "Health Assistant" in page.text
        assert "default-src 'self'" in page.headers["content-security-policy"]
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/ui/", follow_redirects=False).headers["location"] == "/"

        r = c.post("/chat", json={"message": "I have chest pain"}).json()
        assert r["kind"] == "emergency" and r["state"]["stage"] == "idle"
        r = c.post("/chat", json={"message": "hello", "state": r["state"]}).json()
        assert r["kind"] == "message" and r["disclaimer"]
        assert c.post("/chat", json={"message": ""}).status_code == 422
        assert c.post("/chat", json={"message": "hi", "state": {"x": "y" * 3000}}).status_code == 422
