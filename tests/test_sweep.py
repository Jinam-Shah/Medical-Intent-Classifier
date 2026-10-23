import json

import pytest

from src.config import load_config, resolve
from src.sweep import BASE_CONFIG, GRID, REPORTS_ROOT, apply_overrides, pick_best, sweep_config


def test_overrides_copy_and_set_nested_keys():
    cfg = {"model": {"hidden_dim": 512}, "seed": 42}
    out = apply_overrides(cfg, {"model.hidden_dim": 128})
    assert out["model"]["hidden_dim"] == 128
    assert cfg["model"]["hidden_dim"] == 512  # original untouched


def test_unknown_override_key_is_an_error():
    with pytest.raises(KeyError):
        apply_overrides({"model": {"hidden_dim": 512}}, {"model.hiden_dim": 128})


def test_every_grid_config_is_valid_and_writes_to_its_own_dirs():
    base = load_config(resolve(BASE_CONFIG))
    dirs = set()
    for name in GRID:
        cfg = sweep_config(base, name)
        assert cfg["data"]["splits_dir"] == base["data"]["splits_dir"]
        assert cfg["output_dir"] != base["output_dir"]
        dirs.add(cfg["output_dir"])
    assert len(dirs) == len(GRID)


def test_pick_best_uses_val_accuracy_then_val_loss():
    results = [
        {"name": "a", "val_accuracy": 0.80, "best_val_loss": 0.9},
        {"name": "b", "val_accuracy": 0.82, "best_val_loss": 1.2},
        {"name": "c", "val_accuracy": 0.82, "best_val_loss": 1.0},
    ]
    assert pick_best(results)["name"] == "c"


def test_tuned_config_matches_the_sweep_winner():
    best = json.loads(resolve(f"{REPORTS_ROOT}/results.json").read_text(encoding="utf-8"))["best"]
    winner = sweep_config(load_config(resolve(BASE_CONFIG)), best)
    tuned = load_config(resolve("configs/bilstm_v2_tuned.yaml"))
    for section in ("seed", "data", "preprocessing", "embeddings", "model", "training"):
        assert tuned[section] == winner[section], section
