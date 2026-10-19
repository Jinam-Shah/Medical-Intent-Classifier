"""Loading intents, text preprocessing, the fixed train/val/test split, and the vocabulary.

Create the split once (it is committed to Git so every run uses the same one):

    python -m src.data --config configs/bilstm.yaml
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import nltk
import torch
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve

PAD, UNK = "<PAD>", "<UNK>"
SPLITS = ("train", "val", "test")

_stemmer = PorterStemmer()
_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _nltk_ready = True


# ----- Text preprocessing -----

def clean_text(text):
    """Lowercase, drop characters other than letters, digits, and basic punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\?\!°\-]", "", text)
    return " ".join(text.split())


def preprocess(text, stemming=True):
    """Clean, tokenize with NLTK, and optionally Porter-stem. Returns a list of tokens."""
    _ensure_nltk()
    tokens = nltk.word_tokenize(clean_text(text))
    if stemming:
        tokens = [_stemmer.stem(t) for t in tokens]
    return tokens


# ----- Loading and splitting -----

def load_intents(path):
    """Return the intents file as a list of {"text", "intent"} rows."""
    with open(path, "r", encoding="utf-8") as f:
        intents = json.load(f)["intents"]
    return [{"text": p, "intent": it["tag"]} for it in intents for p in it["patterns"]]


def deduplicate(rows):
    """Drop repeated patterns, and drop patterns that appear under more than one intent.

    Returns (kept_rows, dropped_conflicts).
    """
    labels_by_text = {}
    for r in rows:
        labels_by_text.setdefault(r["text"].strip().lower(), set()).add(r["intent"])
    conflicts = sorted(t for t, labels in labels_by_text.items() if len(labels) > 1)

    seen, kept = set(), []
    for r in rows:
        key = r["text"].strip().lower()
        if key in conflicts or key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, conflicts


def make_splits(rows, train_frac, val_frac, test_frac, seed):
    """Stratified train/val/test split."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "split fractions must sum to 1"
    labels = [r["intent"] for r in rows]
    train, rest = train_test_split(
        rows, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    rest_labels = [r["intent"] for r in rest]
    val, test = train_test_split(
        rest, test_size=test_frac / (val_frac + test_frac), stratify=rest_labels, random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def write_splits(splits, splits_dir, manifest):
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(splits_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_split(splits_dir, name):
    path = Path(splits_dir) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create the split first: python -m src.data --config configs/bilstm.yaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----- Vocabulary and encoding -----

class Vocabulary:
    def __init__(self, tokens):
        """tokens: ordered list of words, without PAD and UNK."""
        self.itos = [PAD, UNK] + list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=1):
        counts = Counter(t for tokens in token_lists for t in tokens)
        return cls(sorted(t for t, c in counts.items() if c >= min_freq))

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        """Map tokens to ids, then truncate or pad to max_len. Returns (ids, length)."""
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens][:max_len]
        if not ids:  # empty input after cleaning; give the LSTM one UNK step
            ids = [self.stoi[UNK]]
        length = len(ids)
        return ids + [self.stoi[PAD]] * (max_len - length), length

    def to_list(self):
        return list(self.itos[2:])


def encode_texts(texts, vocab, max_len, stemming=True):
    """Encode raw texts to (ids, lengths) tensors."""
    encoded = [vocab.encode(preprocess(t, stemming), max_len) for t in texts]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
    return ids, lengths


def main():
    parser = argparse.ArgumentParser(description="Create the fixed train/val/test split.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    dcfg = cfg["data"]
    intents_path = resolve(dcfg["intents_file"])

    rows = load_intents(intents_path)
    kept, conflicts = deduplicate(rows)
    splits = make_splits(kept, dcfg["train_frac"], dcfg["val_frac"], dcfg["test_frac"], cfg["seed"])

    manifest = {
        "source_file": dcfg["intents_file"],
        # hash the rows, not the file bytes, so CRLF/LF checkouts give the same value
        "source_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "seed": cfg["seed"],
        "patterns_in_source": len(rows),
        "patterns_after_dedup": len(kept),
        "dropped_conflicting_patterns": conflicts,
        "counts": {name: len(split) for name, split in splits.items()},
        "per_intent": {
            name: dict(sorted(Counter(r["intent"] for r in split).items()))
            for name, split in splits.items()
        },
    }
    write_splits(splits, resolve(dcfg["splits_dir"]), manifest)
    print(json.dumps({k: manifest[k] for k in ("patterns_in_source", "patterns_after_dedup",
                                               "dropped_conflicting_patterns", "counts")}, indent=2))


if __name__ == "__main__":
    main()
