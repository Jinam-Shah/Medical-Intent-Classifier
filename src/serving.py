"""Torch-free pieces shared by the PyTorch and ONNX Runtime classifiers."""

import json
from pathlib import Path

import numpy as np

DEFAULT_CHECKPOINT = "outputs/models/bilstm/model.pt"
DEFAULT_ONNX_MODEL = "outputs/models/bilstm/model.onnx"
ONNX_META_FILE = "model_meta.json"  # vocab, labels, preprocessing settings for the ONNX model
SERVING_FILE = "serving.json"  # temperature and threshold, written by `python -m src.threshold`
DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # used only when no serving.json exists


def softmax(logits, temperature=1.0):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_serving_config(model_dir):
    """Return (temperature, confidence_threshold) from serving.json in model_dir, or defaults."""
    path = Path(model_dir) / SERVING_FILE
    serving = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return float(serving.get("temperature", 1.0)), float(serving.get("confidence_threshold",
                                                                   DEFAULT_CONFIDENCE_THRESHOLD))


def load_classifier(backend="pytorch", model_path=None):
    """Return a PyTorch or ONNX Runtime classifier with the same interface.

    Imports are lazy so the ONNX path never imports torch.
    """
    if backend == "pytorch":
        from src.predict import IntentClassifier
        return IntentClassifier(model_path or DEFAULT_CHECKPOINT)
    if backend == "onnx":
        from src.onnx_predict import OnnxIntentClassifier
        return OnnxIntentClassifier(model_path or DEFAULT_ONNX_MODEL)
    if backend == "transformer":
        from src.transformer import TransformerClassifier
        return TransformerClassifier(model_path)
    if backend == "transformer-onnx":
        from src.onnx_transformer import OnnxTransformerClassifier
        return OnnxTransformerClassifier(model_path)
    raise ValueError(f"unknown backend {backend!r}; use 'pytorch', 'onnx', 'transformer' or 'transformer-onnx'")


def top_k_prediction(probs, labels, k=3):
    """probs: 1-D array for one text. Returns {"intent", "confidence", "top_k"}."""
    order = np.argsort(probs)[::-1][:k]
    top = [{"intent": labels[i], "confidence": float(probs[i])} for i in order]
    return {"intent": top[0]["intent"], "confidence": top[0]["confidence"], "top_k": top}
