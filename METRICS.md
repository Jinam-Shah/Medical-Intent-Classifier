# Metrics

Every number here was measured, not estimated. Resume claims must come from this file.

**Environment A:** laptop, AMD Ryzen AI 7 350 (8 cores / 16 threads), 15 GB RAM visible to Windows, Windows 11 (build 26300), Python 3.11.9, torch 2.14.1+cpu, CPU only. Package versions are in `requirements.txt`.

Test-set rule: the test split is evaluated once per final model. Every tuning decision uses validation. See `docs/DECISIONS.md` D2.

## Phase 0 baseline: BiLSTM + attention (fixed split, seed 42)

Split: 596 train / 128 val / 128 test (`data/splits/manifest.json`). 16,343,053 parameters.

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-19 | Validation accuracy | 0.9688 (124/128) | A | `python -m src.train --config configs/bilstm.yaml` |
| 2026-10-19 | Validation weighted F1 | 0.9687 | A | same |
| 2026-10-19 | Validation macro F1 | 0.9674 | A | same |
| 2026-10-19 | **Test accuracy** (evaluated once) | **0.8906** (114/128), 95% Wilson CI 0.825 to 0.934 | A | `python -m src.evaluate --split test` |
| 2026-10-19 | **Test weighted F1** | **0.8916** | A | same |
| 2026-10-19 | Test macro F1 | 0.8655 | A | same |
| 2026-10-19 | Best epoch / epochs run | 16 / 46 (early stopping, patience 30) | A | `outputs/models/bilstm/summary.json` |
| 2026-10-19 | Training time | 176 s (machine otherwise idle) | A | wall-clock from `src.train` summary, second run |
| 2026-10-19 | Reproducibility | 2 runs gave bit-identical weights | A | trained twice, compared every state-dict tensor with `torch.equal` |
| 2026-10-19 | Inference latency, single query (preprocessing + model) | p50 9.55 ms, p95 12.13 ms, 8 torch threads | A | `python -m src.benchmark --runs 500` |
| 2026-10-19 | Inference latency, single query, 1 thread | p50 13.74 ms, p95 15.32 ms | A | `python -m src.benchmark --runs 500 --threads 1` |
| 2026-10-19 | Token OOV rate, val / test | 2.7% / 5.6% | A | ad hoc analysis against `reports/bilstm/*/errors.json`, described in `docs/AUDIT.md` |

Per-class results: `reports/bilstm/val/` and `reports/bilstm/test/`.

## Audit: original code at commit `be6922d`, unmodified

Different split from the table above (854 patterns, 129 test), so not directly comparable.

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| (original README) | Test accuracy claimed | 0.9457 | unknown | not reproducible: no seeds |
| 2026-10-19 | Test accuracy, rerun 2 | 0.9612 (124/129), weighted F1 0.9609 | A | `MPLBACKEND=Agg python train.py` at `be6922d` |
| 2026-10-19 | Test accuracy, rerun 3 | 0.9070 (117/129), weighted F1 0.9067 | A | same command, same code |
| 2026-10-19 | Rerun 1 | crashed at epoch 10 (Tk backend blocked) | A | `python train.py` at `be6922d` |

## Phase 4b: calibration and low-confidence threshold (BiLSTM baseline)

Out-of-scope set: 80 hand-written messages in 4 categories (`data/out_of_scope.json`), split 40 validation / 40 test. Temperature and threshold were chosen on validation; the test split was evaluated once.

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-19 | Temperature (fit on val NLL) | T = 1.849; val NLL 0.1320 → 0.0973 | A | `python -m src.threshold --split val` |
| 2026-10-19 | Chosen threshold (calibrated) | 0.98 | A | same |
| 2026-10-19 | Val, chosen threshold | OOS rejected 0.950 (38/40); in-scope answered 0.836; in-scope accuracy with unsure = wrong 0.836 | A | same |
| 2026-10-19 | Val, previous default (0.75, uncalibrated) | OOS rejected 0.125; in-scope answered 0.969 | A | same |
| 2026-10-19 | AUROC in-scope vs out-of-scope, val / test | 0.927 / 0.838 | A | max calibrated softmax as the score, `src.threshold` |
| 2026-10-19 | **Test, chosen threshold** (evaluated once) | **OOS rejected 0.825 (33/40)**; in-scope answered 0.828; **accuracy when answered 0.962**; in-scope accuracy with unsure = wrong 0.797 | A | `python -m src.threshold --split test` |
| 2026-10-19 | Test, previous default (0.75, uncalibrated) | OOS rejected 0.100; in-scope answered 0.953 | A | same |
| 2026-10-19 | Test, no threshold | OOS rejected 0.000; in-scope accuracy 0.891 | A | same |

Details: `reports/bilstm/threshold/{val,test}.json` and `val_curve.png`.
