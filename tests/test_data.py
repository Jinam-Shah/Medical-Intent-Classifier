from collections import Counter

from src.config import load_config, resolve
from src.data import (
    PAD,
    UNK,
    Vocabulary,
    clean_text,
    deduplicate,
    encode_texts,
    load_split,
    make_splits,
    preprocess,
)

CFG = load_config(resolve("configs/bilstm.yaml"))
SPLITS_DIR = resolve(CFG["data"]["splits_dir"])


def test_clean_text_lowercases_and_strips_symbols():
    assert clean_text("  I HAVE a Fever!!  ") == "i have a fever!!"
    assert clean_text("my head *hurts* :(") == "my head hurts"


def test_preprocess_stems_tokens():
    assert preprocess("I am coughing a lot") == ["i", "am", "cough", "a", "lot"]
    assert preprocess("Coughing", stemming=False) == ["coughing"]


def test_vocabulary_encode_pads_truncates_and_maps_unknowns():
    vocab = Vocabulary.build([["fever", "cough"], ["fever"]])
    assert vocab.itos[:2] == [PAD, UNK]

    ids, length = vocab.encode(["fever", "unseen"], max_len=5)
    assert ids == [vocab.stoi["fever"], vocab.stoi[UNK], 0, 0, 0]
    assert length == 2

    ids, length = vocab.encode(["fever"] * 10, max_len=4)
    assert len(ids) == 4 and length == 4


def test_empty_input_encodes_to_single_unk():
    vocab = Vocabulary(["fever"])
    ids, lengths = encode_texts(["%%%", ""], vocab, max_len=6)
    assert lengths.tolist() == [1, 1]
    assert ids[0, 0].item() == vocab.stoi[UNK]


def test_deduplicate_drops_repeats_and_cross_label_conflicts():
    rows = [
        {"text": "Hi", "intent": "greeting"},
        {"text": "hi ", "intent": "greeting"},
        {"text": "Ciao", "intent": "greeting"},
        {"text": "ciao", "intent": "goodbye"},
    ]
    kept, conflicts = deduplicate(rows)
    assert conflicts == ["ciao"]
    assert kept == [{"text": "Hi", "intent": "greeting"}]


def test_make_splits_is_stratified_and_deterministic():
    rows = [{"text": f"{label} {i}", "intent": label} for label in "abcd" for i in range(20)]
    first = make_splits(rows, 0.7, 0.15, 0.15, seed=1)
    second = make_splits(rows, 0.7, 0.15, 0.15, seed=1)
    assert first == second
    assert set(Counter(r["intent"] for r in first["test"])) == set("abcd")


def test_committed_splits_are_disjoint_and_cover_all_intents():
    splits = {name: load_split(SPLITS_DIR, name) for name in ("train", "val", "test")}
    texts = {name: {r["text"].strip().lower() for r in rows} for name, rows in splits.items()}
    assert not texts["train"] & texts["val"]
    assert not texts["train"] & texts["test"]
    assert not texts["val"] & texts["test"]

    train_labels = {r["intent"] for r in splits["train"]}
    assert len(train_labels) == 12
    assert {r["intent"] for r in splits["test"]} == train_labels
