"""Request and response models."""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

MAX_TEXT_CHARS = 500
MAX_BATCH_SIZE = 64
MAX_STATE_CHARS = 2000

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


class PredictRequest(BaseModel):
    text: Text = Field(examples=["I have had a fever since yesterday"])


class BatchPredictRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_SIZE,
                              examples=[["hello", "my throat hurts"]])


class IntentScore(BaseModel):
    intent: str
    confidence: float


class Prediction(BaseModel):
    text: str
    intent: str = Field(description='Predicted intent; "out_of_scope" when the model says the message is '
                                    'off-topic, "unsure" when confidence is below the threshold.')
    confidence: float = Field(description="Calibrated probability of the top intent.")
    is_confident: bool
    top_k: list[IntentScore]
    message: str | None = Field(None, description="Set when the model is unsure, or on a safety alert.")
    safety_alert: str | None = Field(
        None, description='"emergency" or "crisis" when a red-flag phrase is detected; overrides the model.')
    disclaimer: str


class BatchPrediction(BaseModel):
    results: list[Prediction]


class Health(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_name: str
    backend: str
    labels: list[str]
    num_parameters: int
    confidence_threshold: float
    temperature: float = Field(description="Softmax temperature from calibration on validation.")
    best_epoch: int | None = None
    trained_at: str | None = None
    disclaimer: str


class ChatRequest(BaseModel):
    message: Text = Field(examples=["my stomach has been hurting since yesterday"])
    state: dict[str, Any] | None = Field(None, description="The `state` returned by the previous turn, unchanged.")

    @field_validator("state")
    @classmethod
    def small_state(cls, v):
        if v is not None and len(json.dumps(v)) > MAX_STATE_CHARS:
            raise ValueError(f"state must be at most {MAX_STATE_CHARS} characters of JSON")
        return v


class ChatResponse(BaseModel):
    reply: str = Field(description="Bot reply; may contain **bold** markers.")
    kind: Literal["message", "emergency", "crisis"]
    quick_replies: list[str] = Field(description="Suggested answers to show as buttons.")
    intents: list[str] = Field(description="Intents detected this turn (for logging and display).")
    state: dict[str, Any] = Field(description="Conversation state; send it back with the next message.")
    disclaimer: str
