"""Evaluate a checkpoint on one split and write metrics, a report, and a confusion matrix.

Use the validation split for every tuning decision:

    python -m src.evaluate --split val

Evaluate on the test split only once, for the final chosen model:

    python -m src.evaluate --split test
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; the Tk backend fails on servers, CI, and locked-down Windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import load_config, resolve  # noqa: E402
from src.data import load_split  # noqa: E402
from src.predict import DEFAULT_CHECKPOINT, IntentClassifier  # noqa: E402


def compute_metrics(y_true, y_pred, labels):
    idx = list(range(len(labels)))
    report = classification_report(
        y_true, y_pred, labels=idx, target_names=labels, output_dict=True, zero_division=0
    )
    return {
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0),
        "per_class": {
            name: {k: report[name][k] for k in ("precision", "recall", "f1-score", "support")}
            for name in labels
        },
    }


def plot_confusion_matrix(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def evaluate(clf, rows, split_name, out_dir):
    """Predict on rows, then save metrics.json, classification_report.txt, and confusion_matrix.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_idx = {name: i for i, name in enumerate(clf.labels)}
    y_true = np.array([label_idx[r["intent"]] for r in rows])
    probs = clf.predict_proba([r["text"] for r in rows])
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, clf.labels)
    metrics.update({
        "split": split_name,
        "mean_confidence": float(probs.max(axis=1).mean()),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": clf.metadata,
    })
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_true, y_pred, labels=list(range(len(clf.labels))),
                                   target_names=clf.labels, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    plot_confusion_matrix(y_true, y_pred, clf.labels, out_dir / "confusion_matrix.png",
                          f"Confusion matrix ({split_name}, n={len(rows)})")

    errors = [
        {"text": r["text"], "true": r["intent"], "pred": clf.labels[p], "confidence": float(probs[i, p])}
        for i, (r, p) in enumerate(zip(rows, y_pred, strict=True)) if clf.labels[p] != r["intent"]
    ]
    with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on one split.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    if args.split == "test":
        print("NOTE: evaluating on the TEST split. Do this once, for the final model only; "
              "make every tuning decision on the validation split.")

    clf = IntentClassifier(args.checkpoint)
    rows = load_split(resolve(cfg["data"]["splits_dir"]), args.split)
    out_dir = resolve(cfg["reports_dir"]) / args.split
    metrics, report = evaluate(clf, rows, args.split, out_dir)

    print(report)
    print(f"{args.split}: accuracy={metrics['accuracy']:.4f}  f1_weighted={metrics['f1_weighted']:.4f}  "
          f"f1_macro={metrics['f1_macro']:.4f}  n={metrics['n_samples']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
