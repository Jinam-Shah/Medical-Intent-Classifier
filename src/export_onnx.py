"""Export the trained checkpoint to ONNX and verify it matches PyTorch.

    python -m src.export_onnx

Writes model.onnx and model_meta.json next to the checkpoint (serving.json is already there),
then compares ONNX Runtime and PyTorch logits on every committed split.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config, resolve
from src.data import load_split
from src.predict import IntentClassifier
from src.serving import DEFAULT_CHECKPOINT, ONNX_META_FILE

OPSET = 17


class UnpaddedBiLSTM(nn.Module):
    """The same weights as BiLSTMAttention, run on one unpadded sentence (batch size 1).

    With no padding, packing and the attention mask are no-ops, so this is numerically the
    same model. It avoids pack_padded_sequence, which ONNX export can't represent.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        m = self.model
        outputs, _ = m.lstm(m.embedding(input_ids))
        weights = torch.softmax(m.attention.score(outputs).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return m.classifier(context)


def export(checkpoint_path, onnx_path):
    clf = IntentClassifier(checkpoint_path)
    wrapper = UnpaddedBiLSTM(clf.model).eval()
    example = torch.tensor([[2, 3, 4]], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # deprecation and batch-size notes from the TorchScript exporter
        # dynamo=False: the torch.export-based exporter unrolls the LSTM with a fixed sequence
        # length; the TorchScript exporter emits a native ONNX LSTM op with a dynamic length.
        torch.onnx.export(
            wrapper, (example,), str(onnx_path),
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {1: "seq_len"}},
            opset_version=OPSET, dynamo=False,
        )
    meta = {
        "labels": clf.labels,
        "vocab": clf.vocab.to_list(),
        "max_seq_length": clf.max_seq_length,
        "stemming": clf.stemming,
        "num_parameters": clf.num_parameters,
        "opset": OPSET,
        **clf.metadata,
    }
    (Path(onnx_path).parent / ONNX_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return clf


def compare(torch_clf, onnx_clf, texts):
    a, b = torch_clf.logits(texts), onnx_clf.logits(texts)
    pa, pb = torch_clf.predict_proba(texts), onnx_clf.predict_proba(texts)
    return {
        "n_texts": len(texts),
        "max_abs_logit_diff": float(np.abs(a - b).max()),
        "max_abs_prob_diff": float(np.abs(pa - pb).max()),
        "argmax_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
    }


def main():
    from src.onnx_predict import OnnxIntentClassifier

    parser = argparse.ArgumentParser(description="Export the checkpoint to ONNX and check parity.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    checkpoint = resolve(args.checkpoint)
    onnx_path = checkpoint.with_suffix(".onnx")
    torch_clf = export(checkpoint, onnx_path)
    onnx_clf = OnnxIntentClassifier(onnx_path)

    cfg = load_config(resolve(args.config))
    splits_dir = resolve(cfg["data"]["splits_dir"])
    texts = [r["text"] for name in ("train", "val", "test", "oos_val", "oos_test")
             for r in load_split(splits_dir, name)]
    report = {"onnx_file": str(onnx_path), "size_mb": round(onnx_path.stat().st_size / 2**20, 1),
              "checkpoint_size_mb": round(checkpoint.stat().st_size / 2**20, 1),
              **compare(torch_clf, onnx_clf, texts)}
    print(json.dumps(report, indent=2))
    if report["argmax_agreement"] < 1.0 or report["max_abs_prob_diff"] > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
