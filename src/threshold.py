"""Calibrate confidences and choose the low-confidence threshold on validation data.

1. Temperature scaling: fit one scalar T > 0 that minimises negative log-likelihood of
   softmax(logits / T) on the in-scope validation split. It never changes the predicted
   intent, only how spread out the confidences are.
2. Threshold: on the calibrated confidences, maximise the mean of
     - in-scope accuracy, counting an "unsure" answer as wrong, and
     - out-of-scope rejection rate (share of out-of-scope messages answered "unsure").
   Ties go to the lowest threshold, so the bot answers whenever it reasonably can.

Both rules were fixed before looking at results. T and the threshold are written to
serving.json next to the checkpoint, which IntentClassifier, the API, and the chatbot read.

    python -m src.threshold --split val     # fit T, choose the threshold, write serving.json
    python -m src.threshold --split test    # report once, for the final model
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data import load_split  # noqa: E402
from src.predict import DEFAULT_CHECKPOINT, SERVING_FILE, IntentClassifier, softmax  # noqa: E402

THRESHOLD_GRID = np.round(np.arange(0.20, 1.00, 0.01), 2)
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(100.0), 2001))


def nll(logits, labels, temperature):
    probs = softmax(logits, temperature)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12)))


def fit_temperature(logits, labels):
    """Grid search over log T; deterministic and plenty precise for one scalar."""
    losses = [nll(logits, labels, t) for t in TEMPERATURE_GRID]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def score_threshold(t, in_conf, in_correct, oos_conf):
    answered = in_conf >= t
    in_acc = float(np.mean(answered & in_correct))
    oos_reject = float(np.mean(oos_conf < t))
    return {
        "threshold": float(t),
        "in_scope_accuracy_with_unsure": in_acc,
        "in_scope_coverage": float(np.mean(answered)),
        "in_scope_accuracy_when_answered": float(np.mean(in_correct[answered])) if answered.any() else None,
        "oos_rejection_rate": oos_reject,
        "objective": (in_acc + oos_reject) / 2,
    }


def choose_threshold(in_conf, in_correct, oos_conf, grid=THRESHOLD_GRID):
    curve = [score_threshold(t, in_conf, in_correct, oos_conf) for t in grid]
    best = max(curve, key=lambda c: (round(c["objective"], 9), -c["threshold"]))
    return best, curve


def plot_curve(curve, chosen, temperature, path):
    t = [c["threshold"] for c in curve]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, [c["in_scope_accuracy_with_unsure"] for c in curve], label="in-scope accuracy (unsure = wrong)")
    ax.plot(t, [c["in_scope_coverage"] for c in curve], "--", label="in-scope coverage (answered)")
    ax.plot(t, [c["oos_rejection_rate"] for c in curve], label="out-of-scope rejection")
    ax.plot(t, [c["objective"] for c in curve], "k", lw=2, label="objective (mean)")
    ax.axvline(chosen, color="gray", ls=":", label=f"chosen = {chosen:.2f}")
    ax.set_xlabel(f"calibrated confidence threshold (T = {temperature:.2f})")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Threshold selection on validation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise(name, r):
    return (f"{name:16s} t={r['threshold']:.2f}  in-scope acc (unsure=wrong)={r['in_scope_accuracy_with_unsure']:.4f}  "
            f"coverage={r['in_scope_coverage']:.4f}  OOS rejected={r['oos_rejection_rate']:.4f}  "
            f"objective={r['objective']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate and choose, or report, the low-confidence threshold.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    splits_dir = resolve(cfg["data"]["splits_dir"])
    clf = IntentClassifier(args.checkpoint)
    in_rows, oos_rows = load_split(splits_dir, args.split), load_split(splits_dir, f"oos_{args.split}")

    in_logits = clf.logits([r["text"] for r in in_rows])
    oos_logits = clf.logits([r["text"] for r in oos_rows])
    labels = np.array([clf.labels.index(r["intent"]) for r in in_rows])
    in_correct = in_logits.argmax(axis=1) == labels

    serving_path = Path(resolve(args.checkpoint)).parent / SERVING_FILE
    if args.split == "val":
        temperature = fit_temperature(in_logits, labels)
    else:
        print("NOTE: test split. Report once, with the temperature and threshold chosen on validation.")
        if not serving_path.exists():
            parser.error(f"{serving_path} not found; run with --split val first")
        temperature = clf.temperature

    in_conf = softmax(in_logits, temperature).max(axis=1)
    oos_conf = softmax(oos_logits, temperature).max(axis=1)
    raw_in_conf, raw_oos_conf = softmax(in_logits).max(axis=1), softmax(oos_logits).max(axis=1)

    out_dir = Path(resolve(cfg["reports_dir"])) / "threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split,
        "n_in_scope": len(in_rows),
        "n_out_of_scope": len(oos_rows),
        "temperature": temperature,
        "nll_uncalibrated": nll(in_logits, labels, 1.0),
        "nll_calibrated": nll(in_logits, labels, temperature),
        "auroc_in_vs_out_of_scope": float(roc_auc_score(
            np.r_[np.ones(len(in_conf)), np.zeros(len(oos_conf))], np.r_[in_conf, oos_conf])),
        "no_threshold": score_threshold(0.0, in_conf, in_correct, oos_conf),
        "previous_default_0.75_uncalibrated": score_threshold(0.75, raw_in_conf, in_correct, raw_oos_conf),
    }

    if args.split == "val":
        best, curve = choose_threshold(in_conf, in_correct, oos_conf)
        result["chosen"], result["curve"] = best, curve
        plot_curve(curve, best["threshold"], temperature, out_dir / "val_curve.png")
        serving = {"temperature": temperature, "confidence_threshold": best["threshold"],
                   "chosen_on": "validation split + oos_val, see reports/bilstm/threshold/val.json"}
        serving_path.write_text(json.dumps(serving, indent=2), encoding="utf-8")
        (out_dir / SERVING_FILE).write_text(json.dumps(serving, indent=2), encoding="utf-8")
    else:
        result["chosen"] = score_threshold(clf.confidence_threshold, in_conf, in_correct, oos_conf)

    oos_probs = softmax(oos_logits, temperature)
    result["oos_still_answered"] = sorted(
        ({"text": r["text"], "category": r["category"], "pred": clf.labels[int(p.argmax())],
          "confidence": float(p.max())}
         for r, p in zip(oos_rows, oos_probs, strict=True) if p.max() >= result["chosen"]["threshold"]),
        key=lambda x: -x["confidence"],
    )
    with open(out_dir / f"{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"temperature T={temperature:.3f}  NLL {result['nll_uncalibrated']:.4f} -> {result['nll_calibrated']:.4f}  "
          f"AUROC(in vs OOS)={result['auroc_in_vs_out_of_scope']:.3f}")
    print(summarise("old 0.75 (raw)", result["previous_default_0.75_uncalibrated"]))
    print(summarise("no threshold", result["no_threshold"]))
    print(summarise("chosen", result["chosen"]))
    print(f"OOS still answered: {len(result['oos_still_answered'])}/{len(oos_rows)}")
    print(f"Saved to {out_dir}" + (f" and {serving_path}" if args.split == "val" else ""))


if __name__ == "__main__":
    main()
