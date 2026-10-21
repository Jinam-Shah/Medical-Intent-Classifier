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


V2_MODEL_DIR = "outputs/models/minilm_v2/onnx_fp32"
V2_ARTIFACTS = ["model.onnx", ONNX_META_FILE, SERVING_FILE, "tokenizer.json"]
V2_COMPARED = [("bilstm_v2", "BiLSTM + attention (from scratch)"), ("distilbert_v2", "DistilBERT base, fine-tuned"),
               ("minilm_v2", "MiniLM-L6, fine-tuned (this model)")]


def model_card_v2():
    """Model card for the dataset-v2 MiniLM model, generated from reports/v2/."""
    test = _load("reports/v2/minilm_v2/test_metrics.json")
    meta = _load(f"{V2_MODEL_DIR}/{ONNX_META_FILE}")
    serving = _load(f"{V2_MODEL_DIR}/{SERVING_FILE}")
    c, e = test["classification"], test["end_to_end"]
    clinc = test["by_source"]["clinc (independent)"]
    own = test["by_source"]["own (v1 + authored)"]
    rows = []
    for name, desc in V2_COMPARED:
        v, t = _load(f"reports/v2/{name}/val_metrics.json"), _load(f"reports/v2/{name}/test_metrics.json")
        rows.append(f"| {desc} | {v['num_parameters'] / 1e6:.1f} M | {v['classification']['accuracy']:.3f} | "
                    f"{t['classification']['accuracy']:.3f} | {t['classification']['f1_macro']:.3f} |")
    labels = ", ".join(f"`{x}`" for x in meta["labels"])
    lo, hi = c["accuracy_ci95"]
    temp, thr = serving["temperature"], serving["confidence_threshold"]
    return f"""---
license: apache-2.0
language: en
pipeline_tag: text-classification
library_name: onnx
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets: [clinc/clinc_oos]
tags: [intent-classification, healthcare, chatbot, minilm, onnx]
---

# Medical Intent Classifier v2 (MiniLM-L6, ONNX)

Classifies a short chatbot message into one of {len(meta["labels"])} intents, including an explicit
`out_of_scope` class: {labels}.

> **Not medical advice.** A portfolio and learning project. It doesn't diagnose anything and is not safe for triage.

Fine-tuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(Apache 2.0). Code, data, and the full measurement log: [{GITHUB_URL}]({GITHUB_URL}) (`METRICS.md`).

## Results (dataset v2 test split, n={test['n']}, evaluated once after choosing the model on validation)

| Metric | Value |
|---|---|
| Accuracy, 26 intents | {c['accuracy']:.3f} (95% CI {lo:.3f} to {hi:.3f}) |
| Macro F1 / weighted F1 | {c['f1_macro']:.3f} / {c['f1_weighted']:.3f} |
| Accuracy on CLINC150 test sentences (independent authors, n={clinc['n']}) | {clinc['classification_accuracy']:.3f} |
| Accuracy on project-written test sentences (n={own['n']}) | {own['classification_accuracy']:.3f} |
| End-to-end with low-confidence fallback (T={temp:.3f}, threshold {thr:.2f}) | {e['end_to_end_accuracy']:.3f} |
| Out-of-scope messages rejected | {e['oos_rejection_rate']:.3f} |

Model comparison on the same split:

| Model | Params | Val accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|---|
{chr(10).join(rows)}

## Files

- `model.onnx`: fp32 ONNX graph; inputs `{", ".join(meta["input_names"])}` (int64, `[batch, seq]`), output `logits`
- `tokenizer.json`: the base model's WordPiece tokenizer (use with the `tokenizers` library)
- `{ONNX_META_FILE}`: labels, input names, max length; `{SERVING_FILE}`: calibration temperature and threshold

## Data

- Original project patterns and patterns written for this project (symptom intents; same author as the training data,
  so those test scores may be optimistic).
- [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (Larson et al., 2019, CC BY 3.0) for greetings, thanks,
  goodbyes, capability questions, and out-of-scope queries, keeping CLINC's own train/validation/test assignment.

## Limitations

- One intent per message ("fever and a cough" gets one label). No injury intents ("I broke my leg" is misread).
- Symptom questions ("how long does the flu last?") get the symptom's general advice.
- Emergency phrases are handled by a rule-based check in the app, not by this model.
"""


def publish(repo_id, model_dir=MODEL_DIR, artifacts=None, card=None, private=False, message=None):
    from huggingface_hub import HfApi

    model_dir = resolve(model_dir)
    artifacts = artifacts or PUBLISHED_ARTIFACTS
    missing = [f for f in artifacts if not (model_dir / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing} in {model_dir}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        for name in artifacts:
            shutil.copy2(model_dir / name, staging / name)
        (staging / "README.md").write_text((card or model_card)(), encoding="utf-8")
        # delete_patterns: remove files from an earlier model (e.g. model.pt) that this version doesn't ship
        commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=staging,
                                   commit_message=message or "Upload intent classifier artifacts",
                                   delete_patterns=["*.pt", "*.onnx", "*.json"])
    print(f"Published https://huggingface.co/{repo_id}")
    print(f"Pinned revision (use for deploys): {commit.oid}")
    return commit.oid


def download(repo_id, revision, dest):
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=repo_id, revision=revision,
                                  allow_patterns=sorted(set(SERVING_ARTIFACTS + V2_ARTIFACTS)), local_dir=dest))
    required = V2_ARTIFACTS if (path / "tokenizer.json").exists() else SERVING_ARTIFACTS
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        raise SystemExit(f"{repo_id}@{revision} is missing {missing}")
    print(f"Downloaded {repo_id}@{revision} to {path}")


def main():
    parser = argparse.ArgumentParser(description="Publish or download model artifacts on the Hugging Face Hub.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--repo", required=True)
    p.add_argument("--model", choices=["v1-bilstm", "v2-minilm"], default="v2-minilm")
    p.add_argument("--private", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--repo", required=True)
    d.add_argument("--revision", required=True, help="commit hash printed by publish")
    d.add_argument("--dest", required=True)
    c = sub.add_parser("card", help="print the generated model card")
    c.add_argument("--model", choices=["v1-bilstm", "v2-minilm"], default="v2-minilm")
    args = parser.parse_args()

    if args.command == "publish":
        if args.model == "v2-minilm":
            publish(args.repo, V2_MODEL_DIR, V2_ARTIFACTS, model_card_v2, args.private,
                    "Upload v2 MiniLM intent classifier (26 intents)")
        else:
            publish(args.repo, private=args.private, message="Upload BiLSTM intent classifier artifacts")
    elif args.command == "download":
        download(args.repo, args.revision, args.dest)
    else:
        print(model_card_v2() if args.model == "v2-minilm" else model_card())


if __name__ == "__main__":
    main()
