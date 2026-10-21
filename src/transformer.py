"""Fine-tune a pretrained transformer (DistilBERT, MiniLM) for intent classification, and export it to ONNX.

    python -m src.transformer train  --config configs/minilm_v2.yaml
    python -m src.transformer export --config configs/minilm_v2.yaml

Model selection uses the validation split only (early stopping on validation loss); the test
split is never read here. Runs are logged to MLflow (src/tracking.py).
"""

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve, set_seed
from src.data import load_split
from src.serving import ONNX_META_FILE, load_serving_config, softmax, top_k_prediction
from src.tracking import start_run

TRAINING_META = "training_meta.json"


class TransformerClassifier:
    """PyTorch inference for a fine-tuned checkpoint directory (save_pretrained format)."""

    backend = "transformer"

    def __init__(self, model_dir, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = resolve(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        self.device = device
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        meta_path = model_dir / TRAINING_META
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.max_length = meta.get("max_length", 64)
        self.metadata = {k: meta[k] for k in ("model_name", "base_model", "best_epoch", "trained_at") if k in meta}
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        self.temperature, self.confidence_threshold = load_serving_config(model_dir)

    @torch.no_grad()
    def logits(self, texts, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else "." for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            out.append(self.model(**enc).logits.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def predict_proba(self, texts, batch_size=64):
        return softmax(self.logits(texts, batch_size), self.temperature)

    def predict(self, text, top_k=3):
        return top_k_prediction(self.predict_proba([text])[0], self.labels, top_k)


def _batches(rows, tokenizer, label_idx, max_length, batch_size, shuffle, seed):
    def collate(batch):
        enc = tokenizer([r["text"] for r in batch], padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        enc["labels"] = torch.tensor([label_idx[r["intent"]] for r in batch], dtype=torch.long)
        return enc
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def _evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        out = model(**batch)
        total_loss += out.loss.item() * len(batch["labels"])
        correct += (out.logits.argmax(1) == batch["labels"]).sum().item()
        n += len(batch["labels"])
    return total_loss / n, correct / n


def train(cfg):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed, mcfg, tcfg = cfg["seed"], cfg["model"], cfg["training"]
    set_seed(seed)
    splits_dir = resolve(cfg["data"]["splits_dir"])
    train_rows, val_rows = load_split(splits_dir, "train"), load_split(splits_dir, "val")
    labels = sorted({r["intent"] for r in train_rows})
    label_idx = {name: i for i, name in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"], revision=mcfg.get("revision"))
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["base_model"], revision=mcfg.get("revision"), num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label_idx)

    train_loader = _batches(train_rows, tokenizer, label_idx, mcfg["max_length"], tcfg["batch_size"], True, seed)
    val_loader = _batches(val_rows, tokenizer, label_idx, mcfg["max_length"], 64, False, seed)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": tcfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=tcfg["learning_rate"])
    total_steps = tcfg["max_epochs"] * math.ceil(len(train_rows) / tcfg["batch_size"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    history, stale = [], 0
    with start_run(cfg["run_name"], tags={"model_family": "transformer", "dataset": "v2"}) as run:
        run.params({"base_model": mcfg["base_model"], "max_length": mcfg["max_length"], **tcfg, "seed": seed,
                    "n_train": len(train_rows), "n_val": len(val_rows), "num_labels": len(labels)})
        start = time.perf_counter()
        for epoch in range(1, tcfg["max_epochs"] + 1):
            model.train()
            total, n = 0.0, 0
            for batch in train_loader:
                out = model(**batch)
                optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += out.loss.item() * len(batch["labels"])
                n += len(batch["labels"])
            val_loss, val_acc = _evaluate(model, val_loader)
            history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_acc": val_acc})
            run.metrics({"train_loss": total / n, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"epoch {epoch:3d}  train_loss={total / n:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
            if val_loss < best["val_loss"] - tcfg["early_stopping_min_delta"]:
                best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
                stale = 0
            else:
                stale += 1
                if stale >= tcfg["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}; best epoch was {best['epoch']}")
                    break
        train_seconds = time.perf_counter() - start

        model.load_state_dict(best["state"])
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        meta = {"model_name": cfg["run_name"], "base_model": mcfg["base_model"], "max_length": mcfg["max_length"],
                "best_epoch": best["epoch"], "epochs_run": len(history), "train_seconds": round(train_seconds, 1),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "history": history}
        (out_dir / TRAINING_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        run.metrics({"best_val_loss": best["val_loss"], "best_epoch": best["epoch"], "train_seconds": train_seconds,
                     "num_parameters": meta["num_parameters"]})
        print(json.dumps({k: meta[k] for k in ("best_epoch", "epochs_run", "train_seconds", "num_parameters")}))
    return out_dir


def export_onnx(cfg, quantize=True, opset=17):
    """Export the fine-tuned model to ONNX (optionally int8-quantized) and check it against PyTorch."""
    from transformers import AutoTokenizer

    from src.onnx_transformer import OnnxTransformerClassifier

    model_dir = resolve(cfg["output_dir"])
    onnx_dir = resolve(cfg["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)
    clf = TransformerClassifier(model_dir)
    model = clf.model
    model.config.return_dict = True

    sample = clf.tokenizer(["i have a sore throat", "hello"], padding=True, return_tensors="pt")
    # only the inputs the model uses (DistilBERT has no token_type_ids; BERT-style MiniLM does)
    input_names = [k for k in ("input_ids", "attention_mask", "token_type_ids")
                   if k in sample and k in clf.tokenizer.model_input_names]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(input_names, args, strict=True))).logits

    fp32_path = onnx_dir / "model_fp32.onnx"
    dynamic = {name: {0: "batch", 1: "seq"} for name in input_names}
    dynamic["logits"] = {0: "batch"}
    torch.onnx.export(Wrapper(model).eval(), tuple(sample[k] for k in input_names), str(fp32_path),
                      input_names=input_names, output_names=["logits"], dynamic_axes=dynamic,
                      opset_version=opset, dynamo=False)
    onnx_file = fp32_path.name
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32_path), str(onnx_dir / "model.onnx"), weight_type=QuantType.QInt8)
        fp32_path.unlink()
        onnx_file = "model.onnx"
    else:
        fp32_path.replace(onnx_dir / "model.onnx")  # replace(): overwrites on Windows too, unlike rename()
        onnx_file = "model.onnx"

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.backend_tokenizer.save(str(onnx_dir / "tokenizer.json"))
    meta = {"labels": clf.labels, "input_names": input_names, "max_length": clf.max_length,
            "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id, "num_parameters": clf.num_parameters,
            "onnx_file": onnx_file, "quantized": quantize, "opset": opset, **clf.metadata}
    (onnx_dir / ONNX_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for name in ("serving.json",):
        if (model_dir / name).exists():
            (onnx_dir / name).write_text((model_dir / name).read_text(encoding="utf-8"), encoding="utf-8")

    onnx_clf = OnnxTransformerClassifier(onnx_dir)
    splits_dir = resolve(cfg["data"]["splits_dir"])
    texts = [r["text"] for s in ("train", "val", "test") for r in load_split(splits_dir, s)]
    a, b = clf.logits(texts), onnx_clf.logits(texts)
    pa, pb = softmax(a), softmax(b)
    report = {"onnx_dir": str(onnx_dir), "quantized": quantize,
              "size_mb": round((onnx_dir / onnx_file).stat().st_size / 2**20, 1), "n_texts": len(texts),
              "argmax_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
              "max_abs_prob_diff": float(np.abs(pa - pb).max())}
    (onnx_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Fine-tune or export a transformer intent classifier.")
    parser.add_argument("command", choices=["train", "export"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    if args.command == "train":
        train(cfg)
    else:
        export_onnx(cfg, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
