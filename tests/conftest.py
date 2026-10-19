import pytest
import torch

from src.model import BiLSTMAttention

LABELS = ["cough", "fever", "greeting"]
VOCAB = ["a", "cough", "fever", "have", "hello", "i"]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A small, randomly initialised checkpoint in the same format src.train writes."""
    torch.manual_seed(0)
    model_kwargs = {"vocab_size": len(VOCAB) + 2, "embedding_dim": 8, "hidden_dim": 8,
                    "num_classes": len(LABELS), "num_layers": 1, "dropout": 0.0, "bidirectional": True}
    model = BiLSTMAttention(**model_kwargs)
    path = tmp_path_factory.mktemp("model") / "model.pt"
    torch.save({
        "model_name": "bilstm_attention",
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "vocab": VOCAB,
        "labels": LABELS,
        "max_seq_length": 10,
        "stemming": True,
        "best_epoch": 1,
        "trained_at": "2026-01-01T00:00:00+00:00",
    }, path)
    return path
