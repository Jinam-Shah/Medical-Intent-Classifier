"""Small BiLSTM hyperparameter sweep on dataset v2, selected on the validation split only.

    python -m src.sweep                      # runs every config not already finished
    python -m src.sweep --only h256_L2_d05   # one config

Each config trains from configs/bilstm_v2.yaml with a few values overridden, and is logged to
MLflow as its own run. All configs share a shorter early-stopping patience than the base config
(the base run's best epoch was 15 of 45), so the comparison is fair and the sweep fits on a laptop.
The test split is never read here: evaluate the winner once with `python -m src.evaluate_v2 test`.
"""

import argparse
import copy
import json

from src.config import load_config, resolve

BASE_CONFIG = "configs/bilstm_v2.yaml"
OUTPUT_ROOT = "outputs/models/bilstm_v2_sweep"
REPORTS_ROOT = "reports/v2/bilstm_sweep"

SHARED = {"training.early_stopping_patience": 10, "training.lr_scheduler_patience": 5}

GRID = {
    "h512_L3_d07": {"model.hidden_dim": 512, "model.num_layers": 3, "model.dropout": 0.7,
                    "training.learning_rate": 0.003},  # the current config
    "h256_L2_d05": {"model.hidden_dim": 256, "model.num_layers": 2, "model.dropout": 0.5,
                    "training.learning_rate": 0.001},
    "h256_L2_d05_lr3": {"model.hidden_dim": 256, "model.num_layers": 2, "model.dropout": 0.5,
                        "training.learning_rate": 0.003},
    "h256_L1_d03": {"model.hidden_dim": 256, "model.num_layers": 1, "model.dropout": 0.3,
                    "training.learning_rate": 0.001},
    "h128_L1_d05": {"model.hidden_dim": 128, "model.num_layers": 1, "model.dropout": 0.5,
                    "training.learning_rate": 0.001},
    "h128_L2_d03_lr3": {"model.hidden_dim": 128, "model.num_layers": 2, "model.dropout": 0.3,
                        "training.learning_rate": 0.003},
}


def apply_overrides(cfg, overrides):
    """Return a copy of cfg with dotted keys ("model.hidden_dim") set; unknown keys are an error."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"unknown config key: {dotted}")
        node[leaf] = value
    return cfg


def sweep_config(base, name):
    cfg = apply_overrides(base, {**SHARED, **GRID[name]})
    cfg["run_name"] = f"bilstm_v2_sweep-{name}"
    cfg["output_dir"] = f"{OUTPUT_ROOT}/{name}"
    cfg["reports_dir"] = f"{REPORTS_ROOT}/{name}"
    return cfg


def pick_best(results):
    """Highest validation accuracy; ties go to the lower validation loss."""
    return max(results, key=lambda r: (round(r["val_accuracy"], 6), -r["best_val_loss"]))


def main():
    parser = argparse.ArgumentParser(description="BiLSTM hyperparameter sweep (validation only).")
    parser.add_argument("--only", nargs="+", choices=sorted(GRID))
    parser.add_argument("--force", action="store_true", help="retrain configs that already finished")
    args = parser.parse_args()

    from src.train import train  # torch is only needed when actually training

    base = load_config(resolve(BASE_CONFIG))
    for name in args.only or GRID:
        summary_path = resolve(OUTPUT_ROOT) / name / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"skip {name}: already trained ({summary_path})")
            continue
        print(f"=== {name}: {GRID[name]}")
        train(sweep_config(base, name))

    results = []
    for name in GRID:
        summary_path = resolve(OUTPUT_ROOT) / name / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            results.append({"name": name, **GRID[name], **summary})
    if not results:
        return
    best = pick_best(results)
    out = resolve(REPORTS_ROOT) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"shared": SHARED, "selected_on": "val_accuracy", "best": best["name"],
                               "results": results}, indent=2), encoding="utf-8")
    for r in sorted(results, key=lambda r: -r["val_accuracy"]):
        print(f"{r['name']:18s} val_acc={r['val_accuracy']:.4f} val_f1_macro={r['val_f1_macro']:.4f} "
              f"val_loss={r['best_val_loss']:.4f} params={r['parameters']:>10,} "
              f"epochs={r['best_epoch']}/{r['epochs_run']} {r['train_seconds']:.0f}s")
    print(f"best: {best['name']}  (written to {out})")


if __name__ == "__main__":
    main()
