"""Load a trained checkpoint and predict intents.

    python -m src.predict "I have a fever" "my throat hurts"
"""

import argparse
import json

import numpy as np
import torch

from src.config import resolve
from src.data import Vocabulary, encode_texts
from src.model import BiLSTMAttention

DEFAULT_CHECKPOINT = "outputs/models/bilstm/model.pt"


class IntentClassifier:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(resolve(checkpoint_path), map_location=self.device, weights_only=True)
        self.labels = ckpt["labels"]
        self.vocab = Vocabulary(ckpt["vocab"])
        self.max_seq_length = ckpt["max_seq_length"]
        self.stemming = ckpt["stemming"]
        self.metadata = {k: ckpt[k] for k in ("model_name", "best_epoch", "val_metrics", "trained_at") if k in ckpt}

        self.model = BiLSTMAttention(**ckpt["model_kwargs"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        """Return class probabilities, shape (len(texts), num_labels)."""
        probs = []
        for i in range(0, len(texts), batch_size):
            ids, lengths = encode_texts(texts[i:i + batch_size], self.vocab, self.max_seq_length, self.stemming)
            logits = self.model(ids.to(self.device), lengths)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, len(self.labels)))

    def predict(self, text, top_k=3):
        """Return {"intent", "confidence", "top_k": [{"intent", "confidence"}, ...]}."""
        probs = self.predict_proba([text])[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"intent": self.labels[i], "confidence": float(probs[i])} for i in order]
        return {"intent": top[0]["intent"], "confidence": top[0]["confidence"], "top_k": top}


def main():
    parser = argparse.ArgumentParser(description="Predict the intent of one or more sentences.")
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    clf = IntentClassifier(args.checkpoint)
    for text in args.texts:
        print(json.dumps({"text": text, **clf.predict(text)}, indent=2))


if __name__ == "__main__":
    main()
