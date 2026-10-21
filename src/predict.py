"""Load a trained checkpoint and predict intents.

    python -m src.predict "I have a fever" "my throat hurts"
    python -m src.predict --backend onnx "I have a fever"
"""

import argparse
import json

import numpy as np
import torch

from src.config import resolve
from src.data import Vocabulary, encode_texts
from src.model import BiLSTMAttention, count_parameters
from src.serving import (  # noqa: F401  (re-exported for callers that import them from here)
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SERVING_FILE,
    load_serving_config,
    softmax,
    top_k_prediction,
)


class IntentClassifier:
    backend = "pytorch"

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
        self.device = torch.device(device)
        checkpoint_path = resolve(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.labels = ckpt["labels"]
        self.vocab = Vocabulary(ckpt["vocab"])
        self.max_seq_length = ckpt["max_seq_length"]
        self.stemming = ckpt["stemming"]
        self.metadata = {k: ckpt[k] for k in ("model_name", "best_epoch", "val_metrics", "trained_at") if k in ckpt}

        self.model = BiLSTMAttention(**ckpt["model_kwargs"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.num_parameters = count_parameters(self.model)

        # Calibration temperature and threshold, both chosen on validation.
        self.temperature, self.confidence_threshold = load_serving_config(checkpoint_path.parent)

    @torch.no_grad()
    def logits(self, texts, batch_size=64):
        """Return raw (uncalibrated) logits, shape (len(texts), num_labels)."""
        out = []
        for i in range(0, len(texts), batch_size):
            ids, lengths = encode_texts(texts[i:i + batch_size], self.vocab, self.max_seq_length, self.stemming)
            out.append(self.model(ids.to(self.device), lengths).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def predict_proba(self, texts, batch_size=64):
        """Return calibrated class probabilities, shape (len(texts), num_labels)."""
        return softmax(self.logits(texts, batch_size), self.temperature)

    def predict(self, text, top_k=3):
        """Return {"intent", "confidence", "top_k": [{"intent", "confidence"}, ...]}."""
        return top_k_prediction(self.predict_proba([text])[0], self.labels, top_k)


def main():
    from src.serving import load_classifier

    parser = argparse.ArgumentParser(description="Predict the intent of one or more sentences.")
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--backend", choices=["pytorch", "onnx", "transformer", "transformer-onnx"],
                        default="pytorch")
    parser.add_argument("--model-path", help="checkpoint (.pt), ONNX model (.onnx), or transformer model directory")
    args = parser.parse_args()

    clf = load_classifier(args.backend, args.model_path)
    for text in args.texts:
        print(json.dumps({"text": text, **clf.predict(text)}, indent=2))


if __name__ == "__main__":
    main()
