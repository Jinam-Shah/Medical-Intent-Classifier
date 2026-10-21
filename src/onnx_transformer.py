"""ONNX Runtime classifier for the fine-tuned transformers. Torch-free: needs only
onnxruntime, numpy, and the small `tokenizers` library.

The model directory holds model.onnx, tokenizer.json, model_meta.json (labels, input names,
max length), and serving.json (temperature and threshold chosen on validation).
"""

import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.config import resolve
from src.serving import ONNX_META_FILE, load_serving_config, softmax, top_k_prediction


class OnnxTransformerClassifier:
    backend = "transformer-onnx"

    def __init__(self, model_path, num_threads=None):
        model_path = resolve(model_path)
        model_dir = model_path.parent if model_path.suffix == ".onnx" else model_path
        meta = json.loads((model_dir / ONNX_META_FILE).read_text(encoding="utf-8"))
        if "input_names" not in meta or not (model_dir / "tokenizer.json").exists():
            raise RuntimeError(
                f"{model_dir} doesn't contain a transformer model (no input_names / tokenizer.json). "
                "It looks like BiLSTM artifacts: check HF_MODEL_REVISION, or use MODEL_BACKEND=onnx.")
        self.labels = meta["labels"]
        self.input_names = meta["input_names"]
        self.max_length = meta["max_length"]
        self.num_parameters = meta["num_parameters"]
        self.metadata = {k: meta[k] for k in ("model_name", "base_model", "best_epoch", "trained_at",
                                              "quantized") if k in meta}

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.max_length)
        self.tokenizer.enable_padding(pad_id=meta["pad_token_id"], pad_token=meta["pad_token"])

        options = ort.SessionOptions()
        num_threads = num_threads or int(os.getenv("ORT_NUM_THREADS", "0"))
        if num_threads:
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        onnx_file = model_dir / meta.get("onnx_file", "model.onnx")
        self.session = ort.InferenceSession(str(onnx_file), options, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]  # feed exactly what the graph declares
        self.temperature, self.confidence_threshold = load_serving_config(model_dir)

    def _encode(self, texts):
        enc = self.tokenizer.encode_batch([t if t.strip() else "." for t in texts])
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                 "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        return feeds

    def logits(self, texts, batch_size=64):
        out = [self.session.run(["logits"], self._encode(texts[i:i + batch_size]))[0]
               for i in range(0, len(texts), batch_size)]
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def predict_proba(self, texts, batch_size=64):
        return softmax(self.logits(texts, batch_size), self.temperature)

    def predict(self, text, top_k=3):
        return top_k_prediction(self.predict_proba([text])[0], self.labels, top_k)
