"""ONNX Runtime classifier with the same interface as src.predict.IntentClassifier.

Imports no torch, so the serving image only needs onnxruntime, numpy, and nltk.
"""

import json
import os

import numpy as np
import onnxruntime as ort

from src.config import resolve
from src.data import Vocabulary, encode_ids
from src.serving import DEFAULT_ONNX_MODEL, ONNX_META_FILE, load_serving_config, softmax, top_k_prediction


class OnnxIntentClassifier:
    backend = "onnx"

    def __init__(self, model_path=DEFAULT_ONNX_MODEL, num_threads=None):
        model_path = resolve(model_path)
        meta = json.loads((model_path.parent / ONNX_META_FILE).read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        self.vocab = Vocabulary(meta["vocab"])
        self.max_seq_length = meta["max_seq_length"]
        self.stemming = meta["stemming"]
        self.num_parameters = meta["num_parameters"]
        self.metadata = {k: meta[k] for k in ("model_name", "best_epoch", "trained_at") if k in meta}

        options = ort.SessionOptions()
        num_threads = num_threads or int(os.getenv("ORT_NUM_THREADS", "0"))
        if num_threads:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

        self.temperature, self.confidence_threshold = load_serving_config(model_path.parent)

    def logits(self, texts, batch_size=None):
        """Raw logits, shape (len(texts), num_labels). The graph takes one unpadded sentence at a time."""
        out = [
            self.session.run(None, {"input_ids": np.array([encode_ids(t, self.vocab, self.max_seq_length,
                                                                         self.stemming)], dtype=np.int64)})[0]
            for t in texts
        ]
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def predict_proba(self, texts, batch_size=None):
        return softmax(self.logits(texts), self.temperature)

    def predict(self, text, top_k=3):
        return top_k_prediction(self.predict_proba([text])[0], self.labels, top_k)
