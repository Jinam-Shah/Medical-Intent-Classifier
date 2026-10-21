import json

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from src.evaluate_v2 import choose, end_to_end, wilson  # noqa: E402
from src.onnx_transformer import OnnxTransformerClassifier  # noqa: E402
from src.transformer import TRAINING_META, TransformerClassifier, export_onnx  # noqa: E402

LABELS = ["fever", "greeting", "out_of_scope"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "have", "a", "fever", "hello", "hi", "weather", "the", "?"]


@pytest.fixture(scope="module")
def tiny_transformer(tmp_path_factory):
    """A tiny randomly initialised BERT saved like a fine-tuned checkpoint (no download)."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny_bert")
    (root / "vocab.txt").write_text("\n".join(VOCAB), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"), do_lower_case=True)
    config = BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, num_labels=len(LABELS),
                        id2label=dict(enumerate(LABELS)), label2id={n: i for i, n in enumerate(LABELS)})
    import torch
    torch.manual_seed(0)
    model_dir = root / "hf"
    BertForSequenceClassification(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / TRAINING_META).write_text(json.dumps({"max_length": 16, "model_name": "tiny"}), encoding="utf-8")
    return root


@pytest.mark.parametrize("quantize", [False, True])
def test_onnx_export_matches_pytorch(tiny_transformer, quantize):
    cfg = {"output_dir": str(tiny_transformer / "hf"), "onnx_dir": str(tiny_transformer / f"onnx_{quantize}"),
           "data": {"splits_dir": "data/v2/splits"}}
    report = export_onnx(cfg, quantize=quantize)
    assert report["argmax_agreement"] >= (0.99 if quantize else 1.0)
    if not quantize:
        assert report["max_abs_prob_diff"] < 1e-4

    onnx_clf = OnnxTransformerClassifier(tiny_transformer / f"onnx_{quantize}")
    torch_clf = TransformerClassifier(tiny_transformer / "hf")
    texts = ["i have a fever", "hello", "", "the weather ?" * 20]  # empty and over-length inputs
    assert onnx_clf.predict_proba(texts).shape == (4, 3)
    assert onnx_clf.labels == torch_clf.labels == LABELS
    assert onnx_clf.predict("hi")["intent"] in LABELS


def test_end_to_end_scoring():
    labels = ["fever", "out_of_scope"]
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    conf = np.array([0.9, 0.9, 0.9, 0.4])
    r = end_to_end(pred, conf, true, labels, threshold=0.5)
    # row0 correct; row1 in-scope but said out_of_scope -> wrong; row2 OOS rejected -> correct;
    # row3 OOS predicted fever but below threshold -> rejected -> correct
    assert r["end_to_end_accuracy"] == 0.75
    assert r["oos_rejection_rate"] == 1.0 and r["in_scope_accuracy_with_unsure"] == 0.5


def test_choose_fits_temperature_and_threshold():
    rng = np.random.default_rng(0)
    labels = ["a", "b", "out_of_scope"]
    true = np.arange(300) % 3
    noisy = np.where(rng.random(300) < 0.2, rng.integers(0, 3, 300), true)  # 20% label noise
    rows = [{"text": str(i), "intent": labels[t]} for i, t in enumerate(noisy)]
    logits = rng.normal(size=(300, 3)) * 0.5
    logits[np.arange(300), true] += 3
    temperature, threshold, extra = choose(rows, logits * 4, labels)  # 4x overconfident
    assert temperature > 1.5
    assert 0.0 <= threshold < 1.0 and extra["nll_calibrated"] <= extra["nll_uncalibrated"]


def test_wilson_interval():
    lo, hi = wilson(90, 100)
    assert 0.82 < lo < 0.9 < hi < 0.95
