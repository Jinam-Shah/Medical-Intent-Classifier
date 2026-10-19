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
