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
