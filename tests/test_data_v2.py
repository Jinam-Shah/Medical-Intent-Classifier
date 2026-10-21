import json
from collections import Counter

import pytest

from src.config import load_config, resolve
from src.data import load_split
from src.data_v2 import drop_cross_source_conflicts, map_clinc, map_clinc_greeting, merge_intents
from src.safety import red_flag

CFG = load_config(resolve("configs/bilstm_v2.yaml"))
SPLITS_DIR = resolve(CFG["data"]["splits_dir"])


@pytest.mark.parametrize("text, intent", [
    ("hey there!", "greeting"), ("hello, friend", "greeting"), ("good to see you", "greeting"),
    ("how have you been", "how_are_you"), ("are you doing alright", "how_are_you"), ("how's it going", "how_are_you"),
])
def test_clinc_greeting_split(text, intent):
    assert map_clinc_greeting(text) == intent


def test_map_clinc_keeps_split_and_excludes_health_adjacent_intents():
    clinc = {"train": [("thanks a ton", "thank_you"), ("do i need a flu shot", "vaccines"),
                       ("what's the dow at", "oos"), ("book a flight", "book_flight")]}
    rows = map_clinc(clinc, seed=0)["train"]
    assert {(r["text"], r["intent"]) for r in rows} == {
        ("thanks a ton", "thanks"), ("what's the dow at", "out_of_scope"), ("book a flight", "out_of_scope")}
    assert all(r["source"] == "clinc" for r in rows)


def test_merge_intents_keeps_v1_responses_and_adds_new_intents():
    v1 = {"intents": [{"tag": "fever", "patterns": ["hot"], "responses": ["rest"]}]}
    authored = {"intents": [{"tag": "fever", "patterns": ["burning up"], "responses": []},
                            {"tag": "toothache", "patterns": ["tooth hurts"], "responses": ["see a dentist"]}]}
    merged = {it["tag"]: it for it in merge_intents(v1, authored)}
    assert merged["fever"] == {"tag": "fever", "patterns": ["hot", "burning up"], "responses": ["rest"]}
    assert merged["toothache"]["responses"] == ["see a dentist"]


def test_cross_source_conflicts_dropped_and_duplicates_kept_in_held_out_split():
    splits = {"train": [{"text": "Hi", "intent": "greeting", "source": "v1"},
                        {"text": "yo", "intent": "greeting", "source": "v1"}],
              "val": [{"text": "hi", "intent": "how_are_you", "source": "clinc"}],
              "test": [{"text": "YO", "intent": "greeting", "source": "clinc"}]}
    cleaned, dropped = drop_cross_source_conflicts(splits)
    assert cleaned == {"test": [{"text": "YO", "intent": "greeting", "source": "clinc"}], "val": [], "train": []}
    assert len(dropped) == 3


def test_committed_v2_splits_are_disjoint_and_complete():
    splits = {s: load_split(SPLITS_DIR, s) for s in ("train", "val", "test")}
    texts = {s: {r["text"].strip().lower() for r in rows} for s, rows in splits.items()}
    assert not texts["train"] & texts["val"] and not texts["train"] & texts["test"] and not texts["val"] & texts["test"]
    intents = {r["intent"] for r in splits["train"]}
    assert len(intents) == 26 and "out_of_scope" in intents
    assert {r["intent"] for r in splits["test"]} == intents
    assert Counter(r["source"] for r in splits["test"])["clinc"] > 0


def test_every_v2_intent_has_responses():
    intents = json.loads(resolve(CFG["data"]["intents_file"]).read_text(encoding="utf-8"))["intents"]
    assert len(intents) == 26 and all(it["responses"] for it in intents)


def test_training_data_has_no_unintended_red_flags():
    """Only genuinely urgent breathing phrases from v1 may trigger the emergency rule."""
    flagged = {r["text"] for s in ("train", "val", "test") for r in load_split(SPLITS_DIR, s) if red_flag(r["text"])}
    assert flagged <= {"I can't breathe properly", "Struggling to breathe"}
