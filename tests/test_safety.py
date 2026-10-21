import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.safety import red_flag


@pytest.mark.parametrize("text", [
    "I have chest pain", "my chest hurts", "pain in my chest when I walk", "I think I'm having a heart attack",
    "I can't breathe", "I cannot breathe properly", "she fainted", "he is having a seizure",
    "bleeding won't stop", "my face is drooping", "my throat is swelling",
])
def test_emergencies(text):
    assert red_flag(text)[0] == "emergency"


@pytest.mark.parametrize("text", ["I want to kill myself", "thinking about suicide", "I want to die"])
def test_crisis(text):
    assert red_flag(text)[0] == "crisis"


@pytest.mark.parametrize("text", [
    "I have a sore throat", "my head hurts", "I have a cough", "hard to breathe through my nose",
    "I have a fever", "thanks", "I'm short of breath after running",
    "I can't breathe through my nose", "can't breathe through nose",
])
def test_routine_messages_are_not_flagged(text):
    assert red_flag(text) is None


def test_api_reports_safety_alert(tiny_checkpoint):
    with TestClient(create_app(model_path=tiny_checkpoint, confidence_threshold=0.0)) as c:
        body = c.post("/predict", json={"text": "I have chest pain"}).json()
        assert body["safety_alert"] == "emergency" and "emergency" in body["message"]
        assert c.post("/predict", json={"text": "I have a fever"}).json()["safety_alert"] is None
