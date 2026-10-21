# Design decisions

Short records of the significant choices in this project: the problem, the options, the choice, and why.

---

## D1. Commit a fixed train/validation/test split to Git

**Problem.** The original code re-split the data inside every training run. Only the split itself was seeded; vocabulary, Word2Vec, and weight initialization were not, so two runs gave different models, and nothing on disk recorded which sentences were in the test set.

**Options.** (a) Keep splitting at train time with a seed. (b) Split once and commit the split files.

**Choice.** (b). `python -m src.data` writes `data/splits/{train,val,test}.jsonl` plus a `manifest.json` (seed, counts per intent, and a hash of the source rows).

**Why.** Anyone can see exactly which 128 sentences are the test set, a reviewer can diff them, and later models (DistilBERT in Phase 2, data additions in Phase 3) are guaranteed to be scored on the same sentences. With a seed alone, adding one pattern to the source file would silently reshuffle everything.

## D2. Validation for every decision; the test set is read once per final model

**Problem.** With 128 test sentences, each one is worth 0.78 percentage points. Picking checkpoints, thresholds, or hyperparameters by looking at test numbers would make the reported test score optimistic.

**Choice.** `src.train` never loads the test split. Early stopping, the hyperparameter sweep (Phase 1), the confusion-matrix review for data improvement (Phase 3), and the confidence threshold (Phase 4) all use validation. `python -m src.evaluate --split test` is run once for the baseline and once for the final model, and both results go in `METRICS.md`.

## D3. Build vocabulary and Word2Vec from the training split only

**Problem.** The original code built the vocabulary and trained Word2Vec on all 854 patterns, including validation and test. No test labels leaked, but the model had embeddings for words that it would otherwise never have seen, which flatters test scores relative to real users.

**Choice.** Vocabulary and Word2Vec are fitted on the training split. Unseen words at evaluation and inference time map to `<UNK>`, as they would in production.

## D4. Mask padding in the attention layer

**Problem.** After `pad_packed_sequence`, padded positions hold zero vectors, and the original attention softmax gave them non-zero weight. In a training batch, a 4-word sentence padded to 12 positions spent part of its attention on padding; at inference, one sentence alone has no padding. The same sentence could therefore get a different prediction depending on what it was batched with (train/serve skew).

**Choice.** Pass a length mask and set padded scores to `-inf` before the softmax. `tests/test_model.py::test_prediction_does_not_depend_on_padding_length` guards this.

## D5. Deep-copy the best weights for early stopping

**Problem.** The original `EarlyStopping` stored `model.state_dict()`, which returns references to the live parameter tensors. The optimizer kept updating them, so "restoring the best model" restored the last epoch instead.

**Choice.** `copy.deepcopy(model.state_dict())` when validation loss improves.

## D6. Drop duplicate and contradictory patterns

**Problem.** `"ciao"` appears under both `greeting` and `goodbye`. Whichever split it lands in, one of the two labels is guaranteed wrong.

**Choice.** `src.data.deduplicate` removes exact (case-insensitive) duplicates and any pattern that appears under more than one intent, and lists dropped patterns in `data/splits/manifest.json`. 854 patterns become 852.

## D7. Checkpoints hold plain data, loaded with `weights_only=True`

**Problem.** The original checkpoint pickled a whole `DataProcessor` object, so loading it required `weights_only=False`, which executes arbitrary pickled code and breaks if the class is renamed.

**Choice.** The checkpoint stores the state dict, model constructor arguments, vocabulary list, label list, and config as plain Python types, and is loaded with `torch.load(..., weights_only=True)`.

## D8. Keep the original hyperparameters for the Phase 0 baseline

**Problem.** The original model has about 16 M parameters for 596 training sentences, which is far larger than needed.

**Choice.** Leave the architecture and hyperparameters unchanged in Phase 0 so the reproduction is a like-for-like comparison. Right-sizing happens in the Phase 1 MLflow sweep, chosen on validation.

## D9. Headless matplotlib backend

**Problem.** The original training crashed at epoch 10 on a Windows machine because matplotlib picked the Tk GUI backend, whose DLL was blocked by Windows Application Control. The same failure happens on servers and CI with no display.

**Choice.** `matplotlib.use("Agg")` in modules that plot.

## D10. FastAPI over Flask for the model service

**Problem.** The original README planned a Flask API. The service needs request validation, typed responses, and API documentation.

**Options.** (a) Flask with hand-written validation and docs. (b) FastAPI.

**Choice.** FastAPI with Pydantic models (`api/schemas.py`).

**Why.** Pydantic rejects bad input (empty or whitespace-only text, over 500 characters, batches over 64, wrong types) with a 422 response before the model runs. The response schema is enforced, and OpenAPI docs at `/docs` are generated from the same models, so they can't drift from the code.

## D11. Load the model once at startup; configure through environment variables

**Choice.** `create_app()` loads the checkpoint in FastAPI's lifespan hook, not per request and not at import time. `MODEL_PATH` and `CONFIDENCE_THRESHOLD` come from environment variables, which is how Docker and Hugging Face Spaces pass configuration.

**Why.** Loading per request would add hundreds of milliseconds. Loading at import time would make the module impossible to import in tests without a trained model. With the factory, tests pass a tiny random checkpoint, so they run in CI in seconds with no model artifact.

## D12. Below the threshold, answer "unsure" but still return the best guess

**Choice.** When the top confidence is below the threshold, `intent` is `"unsure"` and a rephrasing message is added, while `top_k` still lists the model's guesses.

**Why.** For a health-related chatbot, a wrong confident answer is worse than asking again. The client (the chatbot UI) gets a clear signal, and the guesses stay available for logging and debugging. A known limitation is that softmax confidence doesn't catch out-of-scope questions the model is confidently wrong about ("Is this serious?" gets 100% `headache`), so the threshold is a partial safeguard, not a complete one.

## D13. A hand-written out-of-scope set to choose the threshold against

**Problem.** A threshold for answering "unsure" can't be chosen from in-scope data alone: on in-scope data, the best threshold is always 0 (answer everything).

**Choice.** 80 out-of-scope messages in 4 categories (other symptoms, severity and duration questions, clinic admin, non-medical chit-chat), written before any model scores were looked at, and split 50/50 per category into validation and test with the same seed as the main split. A test checks that none of them duplicates a labeled pattern.

**Limitation.** I wrote them myself, so they reflect my guesses about what users might say off-topic. Real user logs would be better.

## D14. Temperature scaling before thresholding

**Problem.** The first threshold search landed at 0.999, the edge of the grid. The model's softmax was saturated: the median out-of-scope message scored 0.9967 and most in-scope messages 0.9999+. The ranking separated them reasonably well (AUROC 0.92), but on a scale where "0.9993 vs 0.9997" decides the outcome, which is fragile and impossible to explain.

**Options.** (a) Search thresholds like 0.9993 directly. (b) Temperature scaling, i.e. divide logits by a scalar T fitted on validation. (c) Train with an explicit out-of-scope class.

**Choice.** (b) now. Temperature scaling (Guo et al., 2017) is one parameter, can't change which intent is predicted, and is fitted on validation NLL, so it doesn't use the out-of-scope data. T = 1.85 confirmed the model was overconfident. (c) is worth revisiting once there is more out-of-scope data.

**Storage.** T and the threshold are written to `serving.json` next to the checkpoint. `IntentClassifier` loads it automatically, so the API, the chatbot, and evaluation all use the same calibrated confidences.

## D15. Threshold objective: equal weight on answering correctly and rejecting off-topic

**Choice.** Maximize the mean of in-scope accuracy (with "unsure" counted as wrong) and out-of-scope rejection rate, fixed before seeing results. Ties go to the lowest threshold.

**Trade-off.** On test, this rejects 82.5% of off-topic messages (vs 10% at the old 0.75) and is right 96.2% of the time when it answers, but it asks for a rephrase on 17% of genuine messages, including some simple ones ("my throat hurts" scores 0.93). Weighting the two terms differently is a product decision, not a modelling one. If it changes, it must be re-chosen on validation, not on test.

## D16. ONNX export through an unpadded single-sentence graph and the TorchScript exporter

**Problem.** The model uses `pack_padded_sequence`, which ONNX can't represent. Also, the default `torch.export`-based exporter (`dynamo=True`) unrolled the LSTM with the example's fixed sequence length: the exported model crashed on any other length.

**Options.** (a) Export with a fixed length of 50 and padding (wasteful, and the backward LSTM would read padding). (b) Export a wrapper that runs one unpadded sentence. (c) Rewrite the LSTM by hand.

**Choice.** (b) with the legacy TorchScript exporter (`dynamo=False`), which emits ONNX's native `LSTM` op with a dynamic sequence length. With no padding, packing and the attention mask are no-ops, so the wrapper is the same function. `src.export_onnx` checks this on all 932 committed sentences and fails if any prediction differs or any probability differs by more than 1e-4 (observed maximum: 6.9e-07).

**Risks.** The TorchScript exporter is deprecated, and torch is pinned. Batch requests are run one sentence at a time through ONNX, which is fine at 2 ms per sentence and a 64-sentence cap.

## D17. Keep the ONNX serving path free of torch

**Choice.** Shared helpers (softmax, serving config, backend selection) live in `src/serving.py`, which doesn't import torch. `src/data.py` and `src/config.py` import torch and scikit-learn only inside the functions that train. The API selects the backend with `MODEL_BACKEND` and imports it lazily.

**Why.** The serving container can then install `onnxruntime` instead of torch, which should make the image much smaller and faster to start (measured in the Docker stage). `tests/test_onnx.py::test_onnx_serving_path_never_imports_torch` runs the API with torch imports blocked, so a stray `import torch` fails CI.

**Why ONNX Runtime is faster here.** Not because of padding: PyTorch running the same unpadded graph is as slow as the padded model. The gain comes from ORT's fused LSTM kernel and from skipping PyTorch's per-operation dispatch overhead, which dominates for a 3-to-10-token input.

## D18. The image packages the evaluated model instead of training during the build

**Problem.** The first Dockerfile trained the model in a build stage. Two things went wrong. First, training on Linux took a different numerical path from the Windows run recorded in `METRICS.md` (Windows early-stopped at epoch 46; the Linux build was still training at epoch 57 with different losses), so the image would have served a model nobody had evaluated. Seeds make a run repeatable on one platform, not across CPU math libraries. Second, the build ran the laptop out of memory and Docker's engine crashed.

**Options.** (a) Train in the build and re-evaluate inside it. (b) Commit the 62 MB model to Git. (c) Copy the locally trained artifacts from the build context. (d) Download artifacts from a model registry during the build.

**Choice.** (c) now, (d) for CI and Hugging Face Spaces in Phase 5, using a free Hugging Face Hub model repo. `.dockerignore` excludes `outputs/` except the four serving files. (b) was rejected because GitHub warns above 50 MB and model binaries bloat history forever.

**Why.** The artifact that is evaluated is the artifact that ships. The test numbers, threshold, and temperature in the image are the ones in `METRICS.md`.

## D19. Slim ONNX Runtime image as the default target

**Choice.** The final (default) stage installs only `requirements-serve.txt`: FastAPI, uvicorn, ONNX Runtime, NumPy, NLTK, PyYAML. It runs as a non-root user, downloads the NLTK tokenizer at build time (the container needs no network at runtime), honours a `PORT` variable (Hugging Face Spaces and Render set one), and has a `HEALTHCHECK` on `/health`. A `serve-pytorch` target is kept for comparison.

**Result.** 604 MB vs 2.34 GB for the PyTorch image, and the Python packages layer is 212 MB vs 1.57 GB. Smaller images pull and cold-start faster on free hosting tiers.

## D20. Host the demo on Render, not Hugging Face Spaces (for now)

**Problem.** The plan assumed free CPU Spaces. As of October 2026, the Hugging Face docs say Gradio and Docker Spaces need a paid plan to create. Free personal accounts can host up to 2 ZeroGPU Gradio Spaces, but only once the account is more than 30 days old, which for this account is after the Nov 7 deadline.

**Options.** (a) Pay for PRO (rejected: free-only rule). (b) Wait for ZeroGPU eligibility. (c) A static Space running the model in the browser with ONNX Runtime Web (would mean re-implementing NLTK tokenization and Porter stemming in JavaScript, with a parity risk). (d) Render's free web service, which runs the existing Docker image.

**Choice.** (d) now, with the API and the Gradio UI in one container. A ZeroGPU Gradio Space can be added after the account turns 30 days old.

**Trade-offs.** Render's free instance has 0.1 CPU and 512 MB, and sleeps after 15 minutes without traffic. A visitor after an idle period waits for Render's wake-up (about a minute) plus about 60 to 75 s of app startup. After that, predictions take about 8 ms.

## D21. One ONNX Runtime thread and no spin-waiting on fractional-CPU hosts

**Problem.** Under a 0.1 CPU limit, ONNX Runtime's defaults (one thread per visible core, here 16, with spin-waiting) burned the CPU quota, so the container was throttled: `/predict` took 2 to 11 s and startup about 3 minutes.

**Choice.** `ORT_NUM_THREADS=1` in the image; `OnnxIntentClassifier` then also sets inter-op threads to 1 and turns spinning off.

**Result.** At 0.1 CPU, `/predict` median went from seconds to 8.7 ms, and startup from 2 min 54 s to 20 s (from the server log line). The input is 3 to 10 tokens, so there's nothing to parallelize anyway.

## D22. Hugging Face Hub as the model registry, with pinned revisions

**Choice.** `python -m src.hub publish` uploads `model.onnx`, `model_meta.json`, `serving.json`, `model.pt`, and a model card to a free HF model repo, and prints the commit hash. Docker builds with `MODEL_SOURCE=hub` download that exact commit (`HF_MODEL_REVISION`), never `main`.

**Why.** Render and CI build from GitHub and don't have the local `outputs/` folder. Pinning the commit means a deploy can't silently pick up a different model. The model card's numbers are generated from the committed `reports/`, so they match `METRICS.md`. (MLflow's registry comes in Phase 1 for experiment tracking; the Hub is for distribution.)

## D23. A rule-based red-flag check in front of the classifier

**Problem.** The classifier answered "I have chest pain" as `headache` with 0.996 confidence. It can't recognize symptoms it was never trained on, and a confidence threshold doesn't catch confident mistakes. For a health chatbot, giving routine self-care advice for an emergency is the worst possible failure.

**Choice.** `src/safety.py` checks a short, explicit list of emergency phrases (chest pain, can't breathe, fainting, seizure, stroke signs, severe bleeding, swelling throat) and crisis phrases (suicide, self-harm) **before** the model runs. On a match, the UI shows an emergency or crisis message instead of the model's reply, and the API sets `safety_alert` and `message`.

**Why rules, not the model.** The behaviour must be predictable, testable phrase by phrase (`tests/test_safety.py`), and independent of retraining. It deliberately errs toward false alarms. It is a safety net, not triage, and the disclaimer stays.

## D24. Chat UI shows replies only

**Choice.** The chat page shows the bot's reply, without the intent label, confidence, or links to the API docs and source code; Gradio's footer links are switched off. The API still returns intent, confidence, and top-3 for developers.

**Why.** It's a demo for people, not a debugging console. Confidence numbers don't help a user and invite over-trust in a model this small.

## D25. Dataset v2: more intents, an explicit out_of_scope class, and an independent data source

**Problem.** The v1 model knew 226 words and 12 intents, so ordinary messages failed: "i have a stomach ache" → `headache` (0.995), because "ache" appeared in training only inside "headache". The v1 data was mostly templates ("I'm experiencing a terrible X"), and with no "other" option, everything off-topic was forced into the nearest keyword.

**Choice.** 26 intents: 12 new symptoms, `capabilities`, and an explicit `out_of_scope` class trained on real off-topic queries. Data comes from three sources, each row marked with its `source`: the v1 patterns; patterns written for this project in varied, natural styles (`data/v2/authored.json`); and CLINC150 (CC BY 3.0) for greetings, thanks, goodbyes, capability questions and out-of-scope queries.

**Why CLINC, and why keep its split.** Its sentences were written by crowd workers who never saw this project. Keeping CLINC's own train/validation/test assignment means 388 of the 673 test sentences are an independent test, reported separately. The symptom-to-diagnosis dataset (gretelai, Apache 2.0) was considered and rejected: its examples are multi-symptom paragraphs labeled by diagnosis, which would add label noise.

**Limitation.** Symptom sentences are still mostly self-written, by the same author for train and test, so those scores may be optimistic.

## D26. Deploy MiniLM-L6 rather than DistilBERT or the BiLSTM

**Options (all trained on the same v2 split).** BiLSTM from scratch (16.6 M parameters), DistilBERT (67 M), MiniLM-L6 (22.7 M; `sentence-transformers/all-MiniLM-L6-v2`, pretrained for sentence similarity).

**Choice, made on validation only.** MiniLM: the best validation accuracy (0.956 vs 0.946 DistilBERT, 0.757 BiLSTM), a third of DistilBERT's size, and the fastest on one CPU thread (2.4 ms p50 vs 10.3 ms DistilBERT, 4.1 ms BiLSTM). DistilBERT and MiniLM are statistically tied (their confidence intervals overlap), so the tie-break is size and speed, which matter on a 0.1-CPU free tier.

**Why the transformers win.** Subword tokenization handles typos and unseen words ("stomache", "tummy", "diarrea"), and pretraining supplies paraphrase knowledge that about 1,800 sentences can't teach a model trained from scratch. The BiLSTM's macro-F1 (0.55 on test) shows it fails on most of the new, smaller intents.

**Test.** Evaluated once after the choice: MiniLM 0.947 (CLINC subset 0.941), DistilBERT 0.936, BiLSTM 0.741.

## D27. Serve fp32 ONNX, not int8

**Trade-off measured on validation.** Dynamic int8 quantization made MiniLM 4× smaller (22 MB vs 87 MB) and 2.8× faster (0.85 ms vs 2.37 ms), but changed 20 of 3,009 predictions and cost 0.8 points of validation accuracy (0.948 vs 0.956). At 0.1 CPU the fp32 model already answers in about 72 ms over HTTP and uses 241 MB of 512 MB, so the speed and size gains don't buy anything users would notice. Accuracy wins.

## D28. MLflow with a local SQLite store, and the Hub for distribution

**Choice.** `src/tracking.py` logs every transformer training run (per-epoch losses), every calibration and test evaluation (metrics plus report artifacts), and the registered model to a local SQLite store in `mlruns/` (ignored by Git). `src/register.py` registers the chosen ONNX model as `medical-intent-classifier`. The Hugging Face Hub remains the distribution channel for deploys, pinned by commit.

**Why local SQLite.** It's free, needs no account, and supports the model registry, which a plain file store doesn't. A shared server (DagsHub) is an option if others need to browse runs. Training still works without MLflow installed.

## D29. Replace Gradio with a hand-built web chat

**Problem.** Gradio is excellent for quick ML demos, but the result looks like a generic ML demo. It also brought about 300 MB of dependencies into the image, slowed startup on the 0.1-CPU free tier, and was hard to restyle (theme overrides, hidden footer links).

**Options.** (a) Keep restyling Gradio. (b) A React/Vite app (build step, Node toolchain, more moving parts). (c) Plain HTML, CSS, and JavaScript served by FastAPI.

**Choice.** (c): three static files in `web/`, with no framework and no CDN. Full control over the design (branded header, welcome screen with suggestions, typing indicator, quick-reply chips, a distinct emergency card with tap-to-call, light and dark themes, mobile layout, accessible roles and labels). Because the page loads nothing from other origins, it's served with a strict Content-Security-Policy (`default-src 'self'`, no inline scripts, `frame-ancestors 'none'`).

**Result** (under Render-like limits, 512 MB / 0.1 CPU): image 990 MB → 703 MB, container start to ready 54 s → 24 s, memory 241 MB → 181 MB.

**Cold starts.** The page pings `/health` on load and shows "Waking up the server…"; chat requests retry for up to two minutes instead of failing, which matters on Render's free tier.

## D30. A stateless conversation engine with the state held by the client

**Problem.** One-shot replies felt basic: the bot couldn't ask how long or how bad a symptom was, and "fever and a cough" got a single label.

**Choice.** `src/dialogue.py` runs a short slot-filling conversation (symptom → duration → severity → advice plus a tailored note), splits multi-symptom messages into clauses, and adds symptoms mentioned mid-conversation. The server is stateless: `POST /chat` returns a small `state` object that the client sends back.

**Why stateless.** No session store to run on a free tier, no user data kept on the server, and nothing breaks when Render restarts or sleeps the service. Because the state comes from the client, it's treated as untrusted: only known keys and values are kept (`clean_state`), and requests over 2 KB of state are rejected.

**Limits.** Duration and severity are parsed with rules (`parse_duration`, `parse_severity`), not a model. The tailored note is generic guidance (for example "severe or longer than a week → see a doctor"), not clinical logic.

## D31. CI that builds and smoke-tests the real image, and Render deploys only after CI passes

**Problem.** A push once reached Render with code expecting the MiniLM model while `render.yaml` still pointed at the BiLSTM files. The deploy crashed at startup (`KeyError: 'input_names'`), and nothing had checked the combination before Render did.

**Choice.** Three GitHub Actions jobs: ruff, the full test suite, and a Docker job that builds the image the way Render does (model from the Hugging Face revision pinned in `render.yaml`), runs it under the free-tier limits (512 MB, 0.1 CPU), and runs the smoke test against it. `render.yaml` uses `autoDeployTrigger: checksPass`. `tests/test_deploy_config.py` adds fast checks for the same failure class: a full commit hash is pinned, the last Dockerfile stage is the served one (Render can't pick a target), and its backend matches the model type.

**Cost.** Free for a public repository. The Docker job reuses layers through the GitHub Actions cache.
