# Phase 0 audit

Date: 2026-10-19. Audited commit: `be6922d` (the state before the Phase 0 changes).

## What existed

| Area | State before Phase 0 |
|---|---|
| Model | PyTorch 3-layer bidirectional LSTM (512 hidden per direction) with attention pooling, 2 fully connected layers, dropout 0.7. 16.35 M parameters. |
| Embeddings | 256-dim Gensim Word2Vec (skip-gram, window 5, 20 epochs), used to initialize a trainable embedding layer. |
| Data | `data/intents_working.json`: 854 patterns, 12 intents (4 conversational, 8 symptoms). A second, unused `data/intents.json` held an older 151-pattern, 27-intent dataset with 48 duplicates. |
| Preprocessing | NLTK: lowercase, strip symbols, `word_tokenize`, Porter stemming, pad or truncate to 50 tokens, packed sequences. |
| Training | Adam (lr 0.003, weight decay 1e-5), ReduceLROnPlateau, gradient clipping at 5.0, early stopping on validation loss (patience 30), 70/15/15 stratified split. |
| Evaluation | Accuracy, macro and weighted F1, per-class report, confusion matrices, plots. |
| Interface | Command-line chatbot (`chatbot.py`, `run.py`). |
| API | `api/__init__.py` placeholder only, no code. |
| Tests, CI, Docker | None. |
| Artifacts in Git | A 63 MB `best_model.pth` plus pickles. |

## Does 94.6% reproduce?

**Not reliably.** The unmodified original code was rerun on the same machine (see `METRICS.md` for the environment):

| Run | Test accuracy | Test weighted F1 | Epochs |
|---|---|---|---|
| Claimed in README | 0.9457 | 0.9448 | 58 |
| Rerun 1 | crashed at epoch 10 (see bug 6) | | |
| Rerun 2 | 0.9612 | 0.9609 | 74 |
| Rerun 3 | 0.9070 | 0.9067 | 46 |

The test set and the code were identical in reruns 2 and 3; only the unseeded random state differed. A 5.4-point spread from randomness alone means one run's number can't be quoted as "the" accuracy. With 129 test sentences, the 95% confidence interval around 94.6% is roughly 89% to 97%.

## Bugs and problems found

1. **Early stopping never restored the best weights.** `EarlyStopping` saved `model.state_dict()`, which holds references to the live tensors. The optimizer kept updating them, so the "best" model was really the final epoch's model. Fixed with `copy.deepcopy`. (DECISIONS D5)
2. **Attention over padding (train/serve skew).** Padded positions got attention weight. In training batches a sentence was padded to the batch's longest sentence; at inference it was alone with no padding, so predictions could differ. Fixed with a mask. (D4)
3. **Vocabulary and Word2Vec fit on all data**, including validation and test. Fixed: both now use the training split only. (D3)
4. **No seeds** for PyTorch, NumPy, or Word2Vec (which also ran with 4 workers, which is non-deterministic). Fixed: the new pipeline produces bit-identical weights across reruns. (D1)
5. **Contradictory label:** `"ciao"` is labeled both `greeting` and `goodbye`. Dropped. (D6)
6. **Training crashed** on this Windows machine at epoch 10: matplotlib picked the Tk backend, and its DLL was blocked by Windows Application Control. The same thing happens on headless servers. Fixed with the `Agg` backend. (D9)
7. **Unsafe checkpoint format:** the checkpoint pickled a `DataProcessor` object and needed `torch.load(weights_only=False)`. Fixed. (D7)
8. **Unpinned dependencies**, including unused ones (Flask, spaCy, Plotly, pandas, torchvision). Now pinned and trimmed.
9. **README placeholders:** the clone URL said `yourusername`. Fixed.
10. **Model size:** 16.35 M parameters for about 600 training sentences. Not changed in Phase 0 so the baseline stays comparable; the Phase 1 sweep addresses it. (D8)

No exact or post-preprocessing duplicates were found between the original train and test splits.

## New baseline (Phase 0 pipeline, same hyperparameters)

Fixed split after deduplication: 596 train, 128 validation, 128 test (`data/splits/`).

| Split | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Validation | 0.9688 | 0.9687 | 0.9674 |
| Test (evaluated once) | 0.8906 | 0.8916 | 0.8655 |

### Why test is 8 points below validation

Mostly unknown words. Because the vocabulary now comes from the training split only, words never seen in training become `<UNK>`, as they would for a real user.

| | Validation | Test |
|---|---|---|
| Share of tokens not in vocabulary | 2.7% | 5.6% |
| Accuracy, sentences with at least one unknown word | 0.857 (n=14) | 0.724 (n=29) |
| Accuracy, sentences where every word is known | 0.982 (n=114) | 0.939 (n=99) |

Single-word inputs like "Hallo", "Hola" and "Ty" become a lone `<UNK>`, and all get the same prediction (`fever`, confidence 0.46). Two things follow:

- The low-confidence fallback in Phase 4 would catch these, since their confidence is far below 0.75.
- A word-level vocabulary can't generalize to unseen words. DistilBERT's subword tokenizer can, which is the main argument for the Phase 2 comparison.

A second weakness shows up in `python chatbot.py test`: questions outside the 12 intents get near-certain predictions ("Is this serious?" → `headache` at 100%; "How long does the flu last?" → `flu` at 100%). Softmax confidence on a closed set of intents isn't a reliable "I don't know" signal, so the Phase 4 fallback needs out-of-scope examples to test against, not only a threshold.

The weakest test classes are `how_are_you` (F1 0.33 on only 4 examples), `greeting` (0.73) and `fever` (0.79; it absorbs most `<UNK>` predictions). Phase 3 uses the **validation** confusion matrix, not these test results, to decide where to add data.

## What works

- The architecture and training recipe are sound, and the CLI chatbot runs.
- Inference is fast: p50 9.6 ms, p95 12.1 ms per query on CPU (`METRICS.md`), well under the claimed 100 ms.
