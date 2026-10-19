"""BiLSTM + attention intent classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive attention pooling over LSTM outputs, ignoring padding positions."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, outputs, mask):
        """
        outputs: (batch, seq_len, dim); mask: (batch, seq_len), True for real tokens.
        Returns context (batch, dim) and weights (batch, seq_len).
        """
        scores = self.score(outputs).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=3, dropout=0.7, bidirectional=True, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained_embeddings, dtype=torch.float32))

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = Attention(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids, lengths, return_attention=False):
        """
        ids: (batch, seq_len) token ids, padded with 0; lengths: (batch,) real lengths.
        Returns logits (batch, num_classes), plus attention weights if requested.
        """
        seq_len = ids.size(1)
        embedded = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=seq_len)

        mask = torch.arange(seq_len, device=ids.device).unsqueeze(0) < lengths.to(ids.device).unsqueeze(1)
        context, weights = self.attention(outputs, mask)
        logits = self.classifier(context)
        return (logits, weights) if return_attention else logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
