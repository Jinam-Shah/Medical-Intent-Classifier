"""Publish the serving artifacts to a Hugging Face Hub model repo, or download a pinned revision.

Log in once in your own terminal (the token is stored in your user profile, never in this repo):
    hf auth login

    python -m src.hub publish --repo <username>/medical-intent-classifier
    python -m src.hub download --repo <username>/medical-intent-classifier --revision <commit> --dest model/

The model card's numbers are read from the committed reports/, so they match METRICS.md.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from src.config import resolve
from src.serving import ONNX_META_FILE, SERVING_FILE

MODEL_DIR = "outputs/models/bilstm"
SERVING_ARTIFACTS = ["model.onnx", ONNX_META_FILE, SERVING_FILE]
PUBLISHED_ARTIFACTS = SERVING_ARTIFACTS + ["model.pt"]
GITHUB_URL = "https://github.com/Jinam-Shah/Medical-Intent-Classifier"


def _load(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def model_card():
    val, test = _load("reports/bilstm/val/metrics.json"), _load("reports/bilstm/test/metrics.json")
    thr_test = _load("reports/bilstm/threshold/test.json")
    serving = _load(f"{MODEL_DIR}/{SERVING_FILE}")
    meta = _load(f"{MODEL_DIR}/{ONNX_META_FILE}")
    chosen, none = thr_test["chosen"], thr_test["no_threshold"]
    n_in, n_oos = thr_test["n_in_scope"], thr_test["n_out_of_scope"]
    temperature, threshold = serving["temperature"], serving["confidence_threshold"]
    acc_none, acc_chosen = none["in_scope_accuracy_when_answered"], chosen["in_scope_accuracy_when_answered"]
    params = meta["num_parameters"]
    labels = ", ".join(f"`{x}`" for x in meta["labels"])
    return f"""---
license: mit
language: en
pipeline_tag: text-classification
library_name: onnx
tags: [intent-classification, healthcare, bilstm, onnx, chatbot]
---

# Medical Intent Classifier (BiLSTM + attention, ONNX)

Classifies a short chatbot message into one of 12 intents: {labels}.

> **Not medical advice.** A portfolio and learning project. It doesn't diagnose anything.

Code, training pipeline, and full measurement log: [{GITHUB_URL}]({GITHUB_URL}) (see `METRICS.md`).

## Results (fixed split, test set evaluated once)

| Split | n | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| Validation | {val['n_samples']} | {val['accuracy']:.3f} | {val['f1_weighted']:.3f} | {val['f1_macro']:.3f} |
| Test | {test['n_samples']} | {test['accuracy']:.3f} | {test['f1_weighted']:.3f} | {test['f1_macro']:.3f} |

Low-confidence fallback on test ({n_in} in-scope, {n_oos} out-of-scope messages),
temperature T = {temperature:.3f} and threshold {threshold:.2f}, both chosen on validation:

| | No fallback | With fallback |
|---|---|---|
| Out-of-scope messages answered "unsure" | {none['oos_rejection_rate']:.1%} | {chosen['oos_rejection_rate']:.1%} |
| In-scope messages answered | {none['in_scope_coverage']:.1%} | {chosen['in_scope_coverage']:.1%} |
| Accuracy when it answers | {acc_none:.1%} | {acc_chosen:.1%} |

## Files

- `model.onnx`: takes `input_ids` (int64, `[1, seq_len]`, one unpadded sentence), returns `logits` `[1, 12]`
- `{ONNX_META_FILE}`: vocabulary, labels, preprocessing settings
- `{SERVING_FILE}`: calibration temperature and confidence threshold
- `model.pt`: the PyTorch checkpoint the ONNX graph was exported from

Preprocessing (NLTK tokenize, Porter stem, vocabulary lookup) lives in the GitHub repo (`src/data.py`,
`src/onnx_predict.py`); the ONNX graph alone is not a complete classifier.

## Limitations

- {params:,} parameters trained on only 596 sentences. Word-level vocabulary: unseen words become `<UNK>`.
- Single intent per message. No severity or duration intents.
- Some out-of-scope messages are still answered confidently, including "I have chest pain when I exercise"
  (answered as `headache`). Not safe for real triage.
"""


def publish(repo_id, private=False):
    from huggingface_hub import HfApi

    model_dir = resolve(MODEL_DIR)
    missing = [f for f in PUBLISHED_ARTIFACTS if not (model_dir / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing} in {model_dir}; run training, src.threshold and src.export_onnx first")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        for name in PUBLISHED_ARTIFACTS:
            shutil.copy2(model_dir / name, staging / name)
        (staging / "README.md").write_text(model_card(), encoding="utf-8")
        commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=staging,
                                   commit_message="Upload BiLSTM intent classifier artifacts")
    print(f"Published https://huggingface.co/{repo_id}")
    print(f"Pinned revision (use for deploys): {commit.oid}")
    return commit.oid


def download(repo_id, revision, dest):
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=SERVING_ARTIFACTS, local_dir=dest)
    missing = [f for f in SERVING_ARTIFACTS if not (Path(path) / f).exists()]
    if missing:
        raise SystemExit(f"{repo_id}@{revision} is missing {missing}")
    print(f"Downloaded {repo_id}@{revision} to {path}")


def main():
    parser = argparse.ArgumentParser(description="Publish or download model artifacts on the Hugging Face Hub.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--repo", required=True)
    p.add_argument("--private", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--repo", required=True)
    d.add_argument("--revision", required=True, help="commit hash printed by publish")
    d.add_argument("--dest", required=True)
    sub.add_parser("card", help="print the generated model card")
    args = parser.parse_args()

    if args.command == "publish":
        publish(args.repo, args.private)
    elif args.command == "download":
        download(args.repo, args.revision, args.dest)
    else:
        print(model_card())


if __name__ == "__main__":
    main()
