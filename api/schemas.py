"""Request and response models."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MAX_TEXT_CHARS = 500
MAX_BATCH_SIZE = 64

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
    intent: str = Field(description='Predicted intent, or "unsure" when confidence is below the threshold.')
    confidence: float = Field(description="Calibrated probability of the top intent.")
    is_confident: bool
    top_k: list[IntentScore]
    message: str | None = Field(None, description="Set when the model is unsure.")
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
