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

## Phase 4d: Docker images

Docker 29.6.2 (Docker Desktop, WSL 2 backend) on Environment A. Sizes are as reported by `docker images` and `docker history`.

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-19 | **Image size, `serve` (ONNX Runtime, no torch)** | **604 MB** (Python packages layer 212 MB, model 65 MB) | A, Docker | `docker build -t medical-intent-classifier:onnx .` then `docker images` / `docker history` |
| 2026-10-19 | Image size, `serve-pytorch` | 2.34 GB (Python packages layer 1.57 GB). Includes the full training stack from `requirements.txt` (gensim, scikit-learn, matplotlib, onnx), not torch alone | A, Docker | `docker build --target serve-pytorch -t medical-intent-classifier:pytorch .` |
| 2026-10-19 | Build time, `serve` (pip cache cold, base image cached) | 65 s | A, Docker | wall clock of the build command |
| 2026-10-19 | Container memory, `serve`, idle after requests | 166.7 MiB | A, Docker | `docker stats --no-stream` |
| 2026-10-19 | Startup to "Application startup complete", `serve`, restart | 3.1 s | A, Docker | `docker restart`, then timestamps from `docker logs -t` |
| 2026-10-19 | Startup, `serve`, first start | 4 min 36 s. **Not representative:** the host had 79 MB of free RAM at the time and was swapping | A, Docker | timestamps from `docker logs -t` |
| 2026-10-19 | Container predictions | Identical to local ONNX Runtime to 5 decimals on 4 spot-check inputs; runs as non-root user `app`; torch not installed; HEALTHCHECK reports `healthy` | A, Docker | `curl` against `/predict`, `/model-info`; `docker exec ... whoami`; `docker inspect` |

## Phase 5 prep: containers under Render free-tier limits (simulated locally)

`docker run --memory 512m --cpus 0.1`, which matches Render's free instance (512 MB RAM, 0.1 CPU). Environment A, Docker 29.6.2. The host had only 177 to 318 MB of free RAM during these runs, so the absolute times may be pessimistic.

| Date | Image / setting | Metric | Value | How it was measured |
|---|---|---|---|---|
| 2026-10-19 | `serve-api`, ONNX Runtime default threads | `/predict` over HTTP, 10 requests | 2.2 to 11.1 s each | `curl -w %{time_total}` against the running container |
| 2026-10-19 | `serve-api`, ONNX Runtime default threads | log "Started server" to "startup complete" | 2 min 54 s | `docker logs -t` timestamps |
| 2026-10-19 | `serve-api`, `ORT_NUM_THREADS=1`, spinning off | `/predict` over HTTP, 20 requests | median 8.7 ms; 19 of 20 under 19 ms; max 4.75 s | same |
| 2026-10-19 | `serve-api`, `ORT_NUM_THREADS=1` | log "Started server" to "startup complete" | 20 s | same |
| 2026-10-19 | `serve-api`, `ORT_NUM_THREADS=1` | idle memory | 158.9 MiB of 512 MiB | `docker stats --no-stream` |
| 2026-10-19 | `serve` (API + Gradio UI), `ORT_NUM_THREADS=1` | `/predict` over HTTP, 20 requests | median 8.0 ms; 19 of 20 under 10.3 ms; max 0.83 s | `curl -w %{time_total}` |
| 2026-10-19 | `serve` | `/ui/` page load | 0.70 s | same |
| 2026-10-19 | `serve` | memory after requests | 229 MiB of 512 MiB; not OOM-killed | `docker stats`, `docker inspect .State.OOMKilled` |
| 2026-10-19 | `serve` | container start to "startup complete" | 60 s (first start), 73 s (restart) | `docker inspect .State.StartedAt` vs `docker logs -t` |
| 2026-10-19 | Image size, `serve` (with Gradio) | | 924 MB | `docker images` |
| 2026-10-19 | Image size, `serve-api` | | 604 MB | `docker images` |

## Phase 5: published model

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-19 | Hugging Face Hub model repo | `Jinamshah/medical-intent-classifier`, revision `f5c65480a954df36f13c507da3c821d1b70339e0` | HF Hub | `python -m src.hub publish --repo Jinamshah/medical-intent-classifier` |
| 2026-10-19 | Published vs evaluated artifacts | SHA-256 identical for `model.onnx`, `model_meta.json`, `serving.json` | A | `python -m src.hub download ... --revision f5c6548...`, then `sha256sum` against `outputs/models/bilstm/` |
| 2026-10-19 | Image built from the Hub revision | builds in 53 s; `/predict` "I have a fever" → `fever` 0.99793, same as the local model | A, Docker | `docker build --build-arg MODEL_SOURCE=hub ...`, then `curl` |

## Phase 5: live deployment (Render free tier)

URL: https://medical-intent-classifier.onrender.com (UI at `/ui/`, API docs at `/docs`). Render free web service, Docker, region virginia, image built from HF revision `f5c6548`. Client: Environment A on a home network, so the latency figures include the internet round trip.

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-21 | End-to-end smoke test (service warm) | 16/16 checks passed: health, model-info (ONNX backend, threshold 0.98, T 1.849), 4 expected predictions including both fallback cases, batch, three 422 validation cases, `/` → `/ui` redirect, UI page, docs | Render free + client A | `python scripts/smoke_test.py https://medical-intent-classifier.onrender.com --requests 30` |
| 2026-10-21 | `/predict` latency over HTTPS, warm, 30 sequential requests | p50 137 ms, p95 176 ms, min 111 ms, max 191 ms | Render free + client A | same command |
| 2026-10-21 | `/ui/` page load, warm | 0.11 s | Render free + client A | same command |

## Dataset v2 and the transformer models (2026-10-21)

Dataset v2: 26 intents, 3,009 sentences (`data/v2/splits/manifest.json`). Train 1,818 / val 518 / test 673. The test split has 388 CLINC150 sentences (independent crowd-worker authors) and 285 project-written sentences (v1 + authored). Environment A, CPU only. The model choice (MiniLM, fp32 ONNX) was fixed on validation before the test split was read; then each compared model was evaluated on test once.

### Validation (used for every decision)

| Model | Accuracy (95% CI) | Macro F1 | Weighted F1 | End-to-end with fallback | T / threshold | Command |
|---|---|---|---|---|---|---|
| BiLSTM v2 | 0.7568 (0.718 to 0.792) | 0.5924 | 0.7527 | 0.7645 | 1.779 / 0.22 | `python -m src.evaluate_v2 calibrate --backend pytorch --model-path outputs/models/bilstm_v2/model.pt --name bilstm_v2` |
| DistilBERT v2 | 0.9459 (0.923 to 0.962) | 0.9368 | 0.9452 | 0.9498 | 0.989 / 0.50 | `... --backend transformer --model-path outputs/models/distilbert_v2/hf --name distilbert_v2` |
| MiniLM v2 (PyTorch) | 0.9556 (0.934 to 0.970) | 0.9427 | 0.9551 | 0.9556 | 1.082 / 0.49 | `... --backend transformer --model-path outputs/models/minilm_v2/hf --name minilm_v2` |
| MiniLM v2, ONNX fp32 | 0.9556 | 0.9427 | 0.9551 | 0.9556 | same | `python -m src.evaluate_v2 val --backend transformer-onnx --model-path outputs/models/minilm_v2/onnx_fp32 --name minilm_v2_onnx_fp32` |
| MiniLM v2, ONNX int8 | 0.9479 (0.925 to 0.964) | 0.9325 | 0.9469 | 0.9479 | same | `... --model-path outputs/models/minilm_v2/onnx --name minilm_v2_onnx` |

Validation by source: BiLSTM 0.675 (own) / 0.855 (CLINC); MiniLM 0.947 (own) / 0.966 (CLINC).

### Test (evaluated once, after the choice)

| Model | Accuracy (95% CI) | Macro F1 | Weighted F1 | Acc. on CLINC (n=388) | Acc. on own (n=285) | End-to-end | OOS rejected | In-scope answered | Command |
|---|---|---|---|---|---|---|---|---|---|
| BiLSTM v2 | 0.7415 (0.707 to 0.773) | 0.5507 | 0.7492 | 0.8325 | 0.6175 | 0.7548 | 0.888 | 0.934 | `python -m src.evaluate_v2 test --backend pytorch --model-path outputs/models/bilstm_v2/model.pt --name bilstm_v2` |
| DistilBERT v2 | 0.9361 (0.915 to 0.952) | 0.9308 | 0.9362 | 0.9433 | 0.9263 | 0.9287 | 0.964 | 0.934 | `... --backend transformer --model-path outputs/models/distilbert_v2/hf --name distilbert_v2` |
| **MiniLM v2 (deployed)** | **0.9465 (0.927 to 0.961)** | **0.9422** | 0.9466 | **0.9407** | 0.9544 | **0.9435** | 0.960 | 0.967 | `... --backend transformer --model-path outputs/models/minilm_v2/hf --name minilm_v2` |
| MiniLM v2, ONNX fp32 | identical to the row above | | | | | | | | `... --backend transformer-onnx --model-path outputs/models/minilm_v2/onnx_fp32 --name minilm_v2_onnx_fp32` |

### Training, export, and speed

| Date | Metric | Value | Environment | How it was measured |
|---|---|---|---|---|
| 2026-10-21 | Training time / best epoch | BiLSTM v2 699 s (best 15 of 45); MiniLM 184 s (best 7 of 10); DistilBERT 382 s (best 4 of 7) | A, CPU; the BiLSTM and MiniLM runs partly overlapped other work | `src.train` / `src.transformer train` summaries |
| 2026-10-21 | Parameters | BiLSTM v2 16,632,859; MiniLM 22,723,226; DistilBERT 66,973,466 | | same |
| 2026-10-21 | ONNX size and parity on all 3,009 sentences | MiniLM fp32 86.8 MB, 100% agreement, max prob diff 3.8e-06; MiniLM int8 22.0 MB, 99.34%; DistilBERT fp32 255.6 MB, 100%; DistilBERT int8 64.3 MB, 99.07%; BiLSTM v2 63.5 MB, 100% | A | `export_report.json` from `src.transformer export` / `src.export_onnx` |
| 2026-10-21 | Single-query latency, 1 thread, 500 runs (p50 / p95 ms) | BiLSTM v2 4.09 / 7.65; MiniLM int8 0.85 / 1.32; **MiniLM fp32 2.37 / 3.24**; DistilBERT int8 3.26 / 4.92; DistilBERT fp32 10.34 / 15.77 | A, CPU otherwise idle | `python -m src.benchmark --config configs/bilstm_v2.yaml --backend <b> --model-path <p> --threads 1 --runs 500` |
| 2026-10-21 | MiniLM v2 image under Render limits (`--memory 512m --cpus 0.1`) | `/predict` over HTTP, 30 requests: p50 72 ms, p95 96 ms, max 98 ms; memory 240.6 MiB; not OOM-killed; container start to ready 54 s | A, Docker 29.6.2 | `docker run --memory 512m --cpus 0.1 ...`, then timed `urllib` requests |
| 2026-10-21 | Image size, `serve` with MiniLM v2 | 990 MB | A, Docker | `docker images` |
| 2026-10-21 | MLflow registry | `medical-intent-classifier` version 1 (ONNX fp32 MiniLM), status READY | A, local SQLite store | `python -m src.register` |
