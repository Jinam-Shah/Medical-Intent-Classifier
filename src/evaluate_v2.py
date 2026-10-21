"""Evaluate, calibrate, and choose the fallback threshold for any v2 model.

    # model comparison on validation (no files written next to the model)
    python -m src.evaluate_v2 val --backend transformer --model-path outputs/models/minilm_v2/hf --name minilm_v2
    # fit temperature + threshold on validation, write serving.json next to the model
    python -m src.evaluate_v2 calibrate --backend transformer --model-path outputs/models/minilm_v2/hf --name minilm_v2
    # once, for the final chosen model
    python -m src.evaluate_v2 test --backend transformer --model-path outputs/models/minilm_v2/hf --name minilm_v2

Backends: pytorch (BiLSTM .pt), onnx (BiLSTM .onnx), transformer (HF dir), transformer-onnx (ONNX dir).

Two views of performance:
  classification  plain 26-way accuracy / macro-F1, no fallback
  end-to-end      with the fallback: an in-scope message is correct only if answered with the right
                  intent; an out-of-scope message is correct if the model says out_of_scope or is unsure
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data import load_split  # noqa: E402
from src.evaluate import plot_confusion_matrix  # noqa: E402
from src.serving import SERVING_FILE, load_classifier, softmax  # noqa: E402
from src.threshold import THRESHOLD_GRID, fit_temperature, nll  # noqa: E402
from src.tracking import start_run  # noqa: E402

OOS = "out_of_scope"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def end_to_end(pred, conf, true, labels, threshold):
    """Per-row correctness with the fallback, plus the components."""
    oos_idx = labels.index(OOS)
    rejected = (pred == oos_idx) | (conf < threshold)
    is_oos = true == oos_idx
    correct = np.where(is_oos, rejected, ~rejected & (pred == true))
    in_acc = float(np.mean(correct[~is_oos])) if (~is_oos).any() else None
    oos_rej = float(np.mean(rejected[is_oos])) if is_oos.any() else None
    answered = ~rejected & ~is_oos
    return {
        "threshold": float(threshold),
        "end_to_end_accuracy": float(np.mean(correct)),
        "in_scope_accuracy_with_unsure": in_acc,
        "in_scope_coverage": float(np.mean(~rejected[~is_oos])) if (~is_oos).any() else None,
        "in_scope_accuracy_when_answered": float(np.mean(pred[answered] == true[answered])) if answered.any() else None,
        "oos_rejection_rate": oos_rej,
        "objective": (in_acc + oos_rej) / 2 if in_acc is not None and oos_rej is not None else None,
        "_correct": correct,
    }


def summarize(rows, logits, labels, temperature, threshold):
    label_idx = {n: i for i, n in enumerate(labels)}
    true = np.array([label_idx[r["intent"]] for r in rows])
    probs = softmax(logits, temperature)
    pred, conf = probs.argmax(1), probs.max(1)
    idx = list(range(len(labels)))
    e2e = end_to_end(pred, conf, true, labels, threshold)
    correct = e2e.pop("_correct")
    out = {
        "n": len(rows),
        "classification": {
            "accuracy": float(accuracy_score(true, pred)),
            "accuracy_ci95": wilson(int((pred == true).sum()), len(rows)),
            "f1_macro": float(f1_score(true, pred, labels=idx, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(true, pred, labels=idx, average="weighted", zero_division=0)),
        },
        "end_to_end": {**e2e, "ci95": wilson(int(correct.sum()), len(rows))},
        "by_source": {},
    }
    sources = np.array([r.get("source", "unknown") for r in rows])
    groups = {"own (v1 + authored)": np.isin(sources, ["v1", "authored"]), "clinc (independent)": sources == "clinc"}
    for name, mask in groups.items():
        if mask.any():
            e = end_to_end(pred[mask], conf[mask], true[mask], labels, threshold)
            e.pop("_correct")
            out["by_source"][name] = {"n": int(mask.sum()), "classification_accuracy":
                                      float(np.mean(pred[mask] == true[mask])), **e}
    report = classification_report(true, pred, labels=idx, target_names=labels, digits=3, zero_division=0)
    errors = [{"text": r["text"], "true": r["intent"], "pred": labels[p], "confidence": float(c),
               "source": r.get("source")} for r, p, c, t in zip(rows, pred, conf, true, strict=True) if p != t]
    return out, report, errors, true, pred


def choose(val_rows, logits, labels):
    """Fit temperature on validation NLL, then the threshold maximising the objective."""
    label_idx = {n: i for i, n in enumerate(labels)}
    true = np.array([label_idx[r["intent"]] for r in val_rows])
    temperature = fit_temperature(logits, true)
    probs = softmax(logits, temperature)
    pred, conf = probs.argmax(1), probs.max(1)
    curve = []
    for t in np.concatenate([[0.0], THRESHOLD_GRID]):
        e = end_to_end(pred, conf, true, labels, t)
        e.pop("_correct")
        curve.append(e)
    best = max(curve, key=lambda c: (round(c["objective"], 9), -c["threshold"]))
    return temperature, best["threshold"], {"nll_uncalibrated": nll(logits, true, 1.0),
                                             "nll_calibrated": nll(logits, true, temperature), "curve": curve}


def main():
    parser = argparse.ArgumentParser(description="Evaluate / calibrate a v2 model.")
    parser.add_argument("mode", choices=["val", "calibrate", "test"])
    parser.add_argument("--backend", required=True, choices=["pytorch", "onnx", "transformer", "transformer-onnx"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True, help="report folder name, e.g. minilm_v2")
    parser.add_argument("--config", default="configs/bilstm_v2.yaml", help="for the splits_dir")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    splits_dir = resolve(cfg["data"]["splits_dir"])
    clf = load_classifier(args.backend, args.model_path)
    model_path = resolve(args.model_path)
    model_dir = model_path if model_path.is_dir() else model_path.parent
    out_dir = resolve(f"reports/v2/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split = "test" if args.mode == "test" else "val"
    rows = load_split(splits_dir, split)
    logits = clf.logits([r["text"] for r in rows])

    if args.mode == "calibrate":
        temperature, threshold, extra = choose(rows, logits, clf.labels)
        serving = {"temperature": temperature, "confidence_threshold": threshold,
                   "chosen_on": f"v2 validation split, see reports/v2/{args.name}/calibration.json"}
        (model_dir / SERVING_FILE).write_text(json.dumps(serving, indent=2), encoding="utf-8")
        (out_dir / SERVING_FILE).write_text(json.dumps(serving, indent=2), encoding="utf-8")
        (out_dir / "calibration.json").write_text(json.dumps({**serving, **extra}, indent=2), encoding="utf-8")
    else:
        if args.mode == "test":
            print("NOTE: test split. Run once, for the final chosen model, with settings chosen on validation.")
        temperature, threshold = clf.temperature, clf.confidence_threshold

    summary, report, errors, true, pred = summarize(rows, logits, clf.labels, temperature, threshold)
    summary.update({"split": split, "model": args.name, "backend": args.backend, "temperature": temperature,
                    "num_parameters": clf.num_parameters,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    tag = "test" if args.mode == "test" else "val"
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{tag}_classification_report.txt").write_text(report, encoding="utf-8")
    (out_dir / f"{tag}_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_confusion_matrix(true, pred, clf.labels, out_dir / f"{tag}_confusion_matrix.png",
                          f"{args.name} confusion matrix ({split}, n={len(rows)})")

    with start_run(f"{args.name}-{args.mode}", tags={"model": args.name, "mode": args.mode, "dataset": "v2"}) as run:
        run.params({"backend": args.backend, "temperature": temperature, "threshold": threshold})
        c, e = summary["classification"], summary["end_to_end"]
        run.metrics({f"{tag}_accuracy": c["accuracy"], f"{tag}_f1_macro": c["f1_macro"],
                     f"{tag}_f1_weighted": c["f1_weighted"], f"{tag}_end_to_end_accuracy": e["end_to_end_accuracy"],
                     f"{tag}_oos_rejection": e["oos_rejection_rate"] or 0,
                     f"{tag}_in_scope_coverage": e["in_scope_coverage"] or 0})
        run.artifacts(out_dir, artifact_path="reports")

    c, e = summary["classification"], summary["end_to_end"]
    print(f"{args.name} [{split}] n={summary['n']}  accuracy={c['accuracy']:.4f} {c['accuracy_ci95']}  "
          f"macro-F1={c['f1_macro']:.4f}  weighted-F1={c['f1_weighted']:.4f}")
    print(f"  end-to-end (T={temperature:.3f}, threshold={threshold:.2f}): {e['end_to_end_accuracy']:.4f} {e['ci95']}  "
          f"OOS rejected={e['oos_rejection_rate']:.4f}  in-scope answered={e['in_scope_coverage']:.4f}  "
          f"accuracy when answered={e['in_scope_accuracy_when_answered']:.4f}")
    for name, s in summary["by_source"].items():
        print(f"  {name:22s} n={s['n']:4d}  classification acc={s['classification_accuracy']:.4f}  "
              f"end-to-end={s['end_to_end_accuracy']:.4f}")
    print(f"Saved to {Path(out_dir)}")


if __name__ == "__main__":
    main()
