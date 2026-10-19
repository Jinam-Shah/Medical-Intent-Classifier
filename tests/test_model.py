import torch

from src.model import BiLSTMAttention


def small_model(num_classes=5):
    torch.manual_seed(0)
    return BiLSTMAttention(vocab_size=20, embedding_dim=8, hidden_dim=16, num_classes=num_classes,
                           num_layers=2, dropout=0.1).eval()


def test_output_shapes():
    model = small_model()
    ids = torch.tensor([[2, 3, 4, 0, 0, 0], [5, 6, 0, 0, 0, 0], [7, 8, 9, 10, 11, 12]])
    lengths = torch.tensor([3, 2, 6])

    logits, weights = model(ids, lengths, return_attention=True)
    assert logits.shape == (3, 5)
    assert weights.shape == (3, 6)


def test_attention_ignores_padding():
    model = small_model()
    ids = torch.tensor([[2, 3, 4, 0, 0, 0]])
    lengths = torch.tensor([3])

    _, weights = model(ids, lengths, return_attention=True)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert torch.all(weights[0, 3:] == 0)


def test_prediction_does_not_depend_on_padding_length():
    """A sentence alone (no padding) and the same sentence in a padded batch must agree."""
    model = small_model()
    alone = model(torch.tensor([[2, 3, 4]]), torch.tensor([3]))
    padded = model(torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]]), torch.tensor([3]))
    assert torch.allclose(alone, padded, atol=1e-6)
