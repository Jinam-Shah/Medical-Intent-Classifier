"""Train the BiLSTM + attention classifier on the fixed training split.

    python -m src.train --config configs/bilstm.yaml

Model selection (early stopping) uses the validation split only. The test split is
never read here; evaluate it once at the end with `python -m src.evaluate --split test`.
"""

import argparse
import copy
import json
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.config import load_config, resolve, set_seed  # noqa: E402
from src.data import Vocabulary, encode_texts, load_split, preprocess  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.model import BiLSTMAttention, count_parameters  # noqa: E402
from src.predict import IntentClassifier  # noqa: E402


def build_embedding_matrix(token_lists, vocab, ecfg, seed):
    """Train Word2Vec on the training sentences and align it to the vocabulary."""
    w2v = Word2Vec(
        sentences=token_lists,
        vector_size=ecfg["dim"],
        window=ecfg["window"],
        min_count=ecfg["min_count"],
        sg=1 if ecfg["skip_gram"] else 0,
        epochs=ecfg["epochs"],
        seed=seed,
        workers=1,  # multiple workers make Word2Vec non-deterministic
    )
    rng = np.random.default_rng(seed)
    matrix = np.zeros((len(vocab), ecfg["dim"]), dtype=np.float32)
    for idx, token in enumerate(vocab.itos):
        if idx == 0:  # PAD stays zero
            continue
        matrix[idx] = w2v.wv[token] if token in w2v.wv else rng.normal(0, 0.1, ecfg["dim"])
    return matrix


def run_epoch(model, loader, criterion, optimizer=None, grad_clip_norm=None):
    """One pass over loader. Trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for ids, lengths, labels in loader:
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for key, ax in (("loss", ax1), ("acc", ax2)):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label=f"train {key}")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label=f"val {key}")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(cfg):
    seed = cfg["seed"]
    set_seed(seed)
    pcfg, tcfg, mcfg = cfg["preprocessing"], cfg["training"], cfg["model"]
    splits_dir = resolve(cfg["data"]["splits_dir"])
    out_dir = resolve(cfg["output_dir"])
    reports_dir = resolve(cfg["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split(splits_dir, "train")
    val_rows = load_split(splits_dir, "val")
    labels = sorted({r["intent"] for r in train_rows})
    label_idx = {name: i for i, name in enumerate(labels)}

    # Vocabulary and embeddings come from the training split only.
    train_tokens = [preprocess(r["text"], pcfg["stemming"]) for r in train_rows]
    vocab = Vocabulary.build(train_tokens)
    embeddings = build_embedding_matrix(train_tokens, vocab, cfg["embeddings"], seed)

    def make_loader(rows, shuffle):
        ids, lengths = encode_texts([r["text"] for r in rows], vocab, pcfg["max_seq_length"], pcfg["stemming"])
        y = torch.tensor([label_idx[r["intent"]] for r in rows], dtype=torch.long)
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(TensorDataset(ids, lengths, y), batch_size=tcfg["batch_size"],
                          shuffle=shuffle, generator=gen)

    train_loader = make_loader(train_rows, shuffle=True)
    val_loader = make_loader(val_rows, shuffle=False)

    model_kwargs = {
        "vocab_size": len(vocab),
        "embedding_dim": cfg["embeddings"]["dim"],
        "hidden_dim": mcfg["hidden_dim"],
        "num_classes": len(labels),
        "num_layers": mcfg["num_layers"],
        "dropout": mcfg["dropout"],
        "bidirectional": mcfg["bidirectional"],
    }
    model = BiLSTMAttention(**model_kwargs, pretrained_embeddings=embeddings)
    print(f"train={len(train_rows)} val={len(val_rows)} vocab={len(vocab)} classes={len(labels)} "
          f"params={count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                                 weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg["lr_scheduler_factor"], patience=tcfg["lr_scheduler_patience"]
    )

    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    history, stale = [], 0
    start = time.perf_counter()
    for epoch in range(1, tcfg["max_epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, tcfg["grad_clip_norm"])
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:4d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best["val_loss"] - tcfg["early_stopping_min_delta"]:
            # deepcopy: state_dict() returns live references that the optimizer keeps updating
            best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
            if stale >= tcfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}; best epoch was {best['epoch']}")
                break
    train_seconds = time.perf_counter() - start

    model.load_state_dict(best["state"])
    checkpoint = {
        "model_name": "bilstm_attention",
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "vocab": vocab.to_list(),
        "labels": labels,
        "max_seq_length": pcfg["max_seq_length"],
        "stemming": pcfg["stemming"],
        "best_epoch": best["epoch"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, reports_dir / "training_history.png")

    clf = IntentClassifier(out_dir / "model.pt")
    val_metrics, _ = evaluate(clf, val_rows, "val", reports_dir / "val")
    summary = {
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_seconds": round(train_seconds, 1),
        "parameters": count_parameters(model),
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_weighted": val_metrics["f1_weighted"],
        "val_f1_macro": val_metrics["f1_macro"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the BiLSTM + attention intent classifier.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    args = parser.parse_args()
    train(load_config(resolve(args.config)))


if __name__ == "__main__":
    main()
