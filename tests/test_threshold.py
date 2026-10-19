import json

import numpy as np
import pytest

from src.predict import SERVING_FILE, IntentClassifier, softmax
from src.threshold import choose_threshold, fit_temperature, nll, score_threshold


def test_softmax_temperature_spreads_but_keeps_argmax():
    logits = np.array([[8.0, 2.0, 1.0], [0.5, 3.0, 0.0]])
    sharp, soft = softmax(logits), softmax(logits, temperature=3.0)
    assert np.allclose(sharp.sum(axis=1), 1) and np.allclose(soft.sum(axis=1), 1)
    assert (sharp.argmax(axis=1) == soft.argmax(axis=1)).all()
    assert (soft.max(axis=1) < sharp.max(axis=1)).all()


def test_fit_temperature_cools_an_overconfident_model():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=300)
    logits = rng.normal(size=(300, 3))
    logits[np.arange(300), labels] += 2.0
    true_t = fit_temperature(logits, labels)
    t = fit_temperature(logits * 5, labels)  # same ranking, 5x too confident
    assert t == pytest.approx(true_t * 5, rel=0.02)
    assert nll(logits * 5, labels, t) < nll(logits * 5, labels, 1.0)


def test_score_threshold_counts_unsure_as_wrong():
    in_conf = np.array([0.9, 0.6, 0.95, 0.3])
    in_correct = np.array([True, True, False, True])
    oos_conf = np.array([0.4, 0.8])
    r = score_threshold(0.5, in_conf, in_correct, oos_conf)
    assert r["in_scope_coverage"] == 0.75
    assert r["in_scope_accuracy_with_unsure"] == 0.5  # 0.9 and 0.6 correct and answered
    assert r["oos_rejection_rate"] == 0.5
    assert r["objective"] == 0.5


def test_choose_threshold_prefers_lowest_on_ties():
    in_conf, in_correct, oos_conf = np.array([0.99]), np.array([True]), np.array([0.1])
    best, _ = choose_threshold(in_conf, in_correct, oos_conf, grid=np.array([0.2, 0.5, 0.9]))
    assert best["threshold"] == 0.2


def test_classifier_reads_serving_file(tiny_checkpoint, tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(tiny_checkpoint.read_bytes())
    plain = IntentClassifier(ckpt)
    (tmp_path / SERVING_FILE).write_text(json.dumps({"temperature": 4.0, "confidence_threshold": 0.66}))
    calibrated = IntentClassifier(ckpt)

    assert (plain.temperature, calibrated.temperature) == (1.0, 4.0)
    assert calibrated.confidence_threshold == 0.66
    texts = ["i have a fever", "hello"]
    assert (plain.predict_proba(texts).argmax(1) == calibrated.predict_proba(texts).argmax(1)).all()
