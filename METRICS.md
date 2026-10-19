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

## Phase 4c: ONNX export and CPU latency (BiLSTM baseline)

Single query, preprocessing + model, 1000 timed runs after 20 warm-up runs, cycling through the 128 validation sentences. Each configuration was run twice; both runs are shown. Machine otherwise idle.

| Date | Backend | Threads | p50 (ms) run 1 / run 2 | p95 (ms) run 1 / run 2 | Environment | Command |
|---|---|---|---|---|---|---|
| 2026-10-19 | PyTorch (padded to 50, packed) | 8 (torch default) | 8.65 / 9.63 | 10.47 / 12.70 | A | `python -m src.benchmark --backend pytorch --runs 1000` |
| 2026-10-19 | PyTorch (padded to 50, packed) | 1 | 16.87 / 17.58 | 21.81 / 21.23 | A | `... --backend pytorch --runs 1000 --threads 1` |
| 2026-10-19 | PyTorch, unpadded graph (same as ONNX export) | 8 | 8.27 / 8.34 | 10.21 / 9.69 | A | `... --backend pytorch-unpadded --runs 1000` |
| 2026-10-19 | PyTorch, unpadded graph | 1 | 17.03 / 16.60 | 22.28 / 21.13 | A | `... --backend pytorch-unpadded --runs 1000 --threads 1` |
| 2026-10-19 | **ONNX Runtime 1.31** | ORT default | **2.02 / 2.14** | **2.63 / 2.81** | A | `python -m src.benchmark --backend onnx --runs 1000` |
| 2026-10-19 | **ONNX Runtime 1.31** | 1 | **3.41 / 3.19** | **5.78 / 5.43** | A | `... --backend onnx --runs 1000 --threads 1` |

Speed-up of ONNX Runtime over PyTorch, p50: 4.3× to 4.5× with default threads, 4.9× to 5.5× on one thread. The unpadded PyTorch rows show that skipping padding explains almost none of it.

Note: an earlier single-thread PyTorch measurement (Phase 0 table, 500 runs) gave p50 13.74 ms. Run-to-run variation on this laptop is real, which is why both runs are listed here.

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-19 | ONNX vs PyTorch parity | 932/932 identical predictions; max abs probability difference 6.9e-07; max abs logit difference 9.5e-06 | A | `python -m src.export_onnx` (all train, val, test, and out-of-scope sentences) |
| 2026-10-19 | Model file size | model.onnx 62.4 MB, model.pt 62.4 MB | A | same |
| 2026-10-19 | Preprocessing alone (clean, tokenize, stem) | p50 0.077 ms | A | 2000 calls to `src.data.preprocess` on one 10-word sentence |
