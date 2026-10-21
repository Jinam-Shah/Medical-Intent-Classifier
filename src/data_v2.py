"""Build dataset v2: 26 intents from three sources, with a fixed train/val/test split.

    python -m src.data_v2

Sources (each row keeps a `source` field):
  v1       data/intents.json, the original 854 patterns (12 intents)
  authored data/v2/authored.json, hand-written patterns: 14 new intents + extra phrasings
  clinc    CLINC150 "plus" (Larson et al., 2019, CC BY 3.0), pinned revision, mapped to
           greeting / how_are_you / thanks / goodbye / capabilities / out_of_scope.
           CLINC's own train/validation/test assignment is kept, so its test rows were
           written by crowd workers who never saw this project: an independent test.

v1 + authored rows are deduplicated and split 70/15/15, stratified, seed 42.
Writes data/v2/intents.json (patterns + responses for serving), data/v2/splits/*.jsonl,
and data/v2/splits/manifest.json.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.config import load_config, resolve
from src.data import deduplicate, make_splits

CLINC_REPO = "clinc/clinc_oos"
CLINC_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
CLINC_CONFIG = "plus"
CLINC_SPLITS = {"train": "train", "validation": "val", "test": "test"}

# How many CLINC rows to take per mapped intent and split (CLINC has 100/20/30 per intent).
CONVERSATIONAL_CAP = {"train": 40, "val": 10, "test": 30}
OOS_CAP = {"train": 150, "val": 40, "test": 100}         # CLINC's explicit out-of-scope queries
OTHER_INTENT_CAP = {"train": 1, "val": 1, "test": 1}     # rows per non-medical CLINC intent, as extra out_of_scope

DIRECT_MAP = {"thank_you": "thanks", "goodbye": "goodbye", "what_can_i_ask_you": "capabilities",
              "are_you_a_bot": "capabilities", "oos": "out_of_scope"}
# CLINC intents near our domain: not used as out_of_scope examples, to avoid confusing labels.
EXCLUDED = {"greeting", "thank_you", "goodbye", "what_can_i_ask_you", "are_you_a_bot", "oos",
            "vaccines", "nutrition_info", "calories", "insurance", "insurance_change"}
HOW_ARE_YOU = re.compile(r"\bhow\b|\bare you\b|\byou (ok|okay|well|alright|doing)\b|\bgoing\b|\bbeen\b|\bthings\b",
                         re.IGNORECASE)


def map_clinc_greeting(text):
    """CLINC's `greeting` mixes "hey there!" with "how have you been"; split them by wording."""
    return "how_are_you" if HOW_ARE_YOU.search(text) else "greeting"


def load_clinc(cache_dir=None):
    """Download the pinned CLINC150 revision. Returns {split: [(text, clinc_intent_name)]}."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    readme = hf_hub_download(CLINC_REPO, "README.md", repo_type="dataset", revision=CLINC_REVISION,
                             cache_dir=cache_dir)
    names = {int(i): n for i, n in re.findall(r"'(\d+)': (\w+)", Path(readme).read_text(encoding="utf-8"))}
    out = {}
    for clinc_split, split in CLINC_SPLITS.items():
        path = hf_hub_download(CLINC_REPO, f"{CLINC_CONFIG}/{clinc_split}-00000-of-00001.parquet",
                               repo_type="dataset", revision=CLINC_REVISION, cache_dir=cache_dir)
        out[split] = [(r["text"], names.get(r["intent"], f"label_{r['intent']}"))
                      for r in pq.read_table(path).to_pylist()]
    return out


def map_clinc(clinc, seed):
    """Map and subsample CLINC rows. Returns {split: [row]} with source="clinc"."""
    rng = random.Random(seed)
    out = {}
    for split, rows in clinc.items():
        by_intent = {}
        for text, name in rows:
            by_intent.setdefault(name, []).append(text)
        mapped = []
        for name, texts in sorted(by_intent.items()):
            texts = sorted(set(texts))
            rng.shuffle(texts)
            if name == "greeting":
                for t in texts[:CONVERSATIONAL_CAP[split] * 2]:
                    mapped.append((t, map_clinc_greeting(t), name))
            elif name == "oos":
                mapped += [(t, "out_of_scope", name) for t in texts[:OOS_CAP[split]]]
            elif name in DIRECT_MAP:
                mapped += [(t, DIRECT_MAP[name], name) for t in texts[:CONVERSATIONAL_CAP[split]]]
            elif name not in EXCLUDED:
                mapped += [(t, "out_of_scope", name) for t in texts[:OTHER_INTENT_CAP[split]]]
        out[split] = [{"text": t, "intent": i, "source": "clinc", "clinc_intent": n} for t, i, n in mapped]
    return out


def merge_intents(v1, authored):
    """v1 patterns + authored patterns; responses from v1 unless the intent is new."""
    merged = {it["tag"]: {"tag": it["tag"], "patterns": list(it["patterns"]), "responses": list(it["responses"])}
              for it in v1["intents"]}
    for it in authored["intents"]:
        entry = merged.setdefault(it["tag"], {"tag": it["tag"], "patterns": [], "responses": []})
        entry["patterns"] += it["patterns"]
        if it["responses"]:
            entry["responses"] = it["responses"]
    missing = [tag for tag, e in merged.items() if not e["responses"]]
    if missing:
        raise SystemExit(f"intents without responses: {missing}")
    return [merged[tag] for tag in sorted(merged)]


def drop_cross_source_conflicts(splits):
    """Remove rows whose text appears with a different label elsewhere, and exact cross-split duplicates."""
    labels = {}
    for rows in splits.values():
        for r in rows:
            labels.setdefault(r["text"].strip().lower(), set()).add(r["intent"])
    conflicts = {t for t, s in labels.items() if len(s) > 1}
    seen, cleaned, dropped = set(), {}, []
    for split in ("test", "val", "train"):  # keep a duplicate in the held-out split, drop it from train
        cleaned[split] = []
        for r in splits[split]:
            key = r["text"].strip().lower()
            if key in conflicts or key in seen:
                dropped.append({"text": r["text"], "intent": r["intent"], "split": split, "source": r["source"]})
                continue
            seen.add(key)
            cleaned[split].append(r)
    return cleaned, dropped


def main():
    parser = argparse.ArgumentParser(description="Build dataset v2.")
    parser.add_argument("--config", default="configs/bilstm_v2.yaml")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    dcfg, seed = cfg["data"], cfg["seed"]

    v1 = json.loads(resolve("data/intents.json").read_text(encoding="utf-8"))
    authored = json.loads(resolve("data/v2/authored.json").read_text(encoding="utf-8"))
    intents = merge_intents(v1, authored)
    resolve(dcfg["intents_file"]).write_text(json.dumps({"intents": intents}, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")

    v1_tags = {it["tag"] for it in v1["intents"]}
    own = [{"text": p, "intent": it["tag"], "source": "v1" if it["tag"] in v1_tags and p in set(
        next(x["patterns"] for x in v1["intents"] if x["tag"] == it["tag"])) else "authored"}
        for it in intents for p in it["patterns"]]
    kept, conflicts = deduplicate(own)
    own_splits = make_splits(kept, dcfg["train_frac"], dcfg["val_frac"], dcfg["test_frac"], seed)

    clinc = map_clinc(load_clinc(), seed)
    combined = {s: own_splits[s] + clinc[s] for s in ("train", "val", "test")}
    splits, dropped = drop_cross_source_conflicts(combined)

    out_dir = resolve(dcfg["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda r: (r["intent"], r["source"], r["text"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": seed,
        "intents": sorted({r["intent"] for r in splits["train"]}),
        "sources": {
            "v1": "data/intents.json",
            "authored": "data/v2/authored.json",
            "clinc": {"repo": CLINC_REPO, "revision": CLINC_REVISION, "config": CLINC_CONFIG,
                      "license": "CC BY 3.0", "citation": "Larson et al., 2019. An Evaluation Dataset for "
                      "Intent Classification and Out-of-Scope Prediction. EMNLP-IJCNLP."},
        },
        "own_rows_hash": hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest(),
        "dropped_conflicting_own_patterns": conflicts,
        "dropped_cross_source_rows": dropped,
        "counts": {s: len(r) for s, r in splits.items()},
        "counts_by_source": {s: dict(sorted(Counter(r["source"] for r in rows).items())) for s, rows in splits.items()},
        "per_intent": {s: dict(sorted(Counter(r["intent"] for r in rows).items())) for s, rows in splits.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "counts_by_source")}, indent=2))
    print(f"{len(manifest['intents'])} intents; dropped {len(dropped)} cross-source rows, "
          f"{len(conflicts)} conflicting own patterns")


if __name__ == "__main__":
    main()
