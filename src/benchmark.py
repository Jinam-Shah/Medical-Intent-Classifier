"""Measure single-query CPU inference latency (preprocessing + model) on validation sentences.

    python -m src.benchmark --backend pytorch --runs 500
    python -m src.benchmark --backend onnx --runs 500
    python -m src.benchmark --backend onnx --threads 1
"""

import argparse
import json
import platform
import time

import numpy as np

from src.config import load_config, resolve
from src.data import load_split
from src.serving import load_classifier


def main():
    parser = argparse.ArgumentParser(description="Single-query latency benchmark.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--backend", choices=["pytorch", "pytorch-unpadded", "onnx"], default="pytorch",
                        help="pytorch-unpadded runs the ONNX export's graph in PyTorch, to separate "
                             "the effect of skipping padding from the effect of ONNX Runtime")
    parser.add_argument("--model-path", help="checkpoint (.pt) or ONNX model (.onnx); default per backend")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None, help="intra-op threads (default: library default)")
    args = parser.parse_args()

    if args.backend.startswith("pytorch"):
        import torch
        if args.threads:
            torch.set_num_threads(args.threads)
        clf = load_classifier("pytorch", args.model_path)
        threads, version = torch.get_num_threads(), f"torch {torch.__version__}"
        if args.backend == "pytorch-unpadded":
            from src.data import encode_ids
            from src.export_onnx import UnpaddedBiLSTM
            wrapper = UnpaddedBiLSTM(clf.model).eval()

            @torch.no_grad()
            def unpadded_logits(texts, batch_size=None):
                return np.concatenate([wrapper(torch.tensor([encode_ids(t, clf.vocab, clf.max_seq_length,
                                                                         clf.stemming)])).numpy() for t in texts])
            clf.logits = unpadded_logits
    else:
        import onnxruntime as ort

        from src.onnx_predict import OnnxIntentClassifier
        clf = OnnxIntentClassifier(args.model_path or "outputs/models/bilstm/model.onnx", num_threads=args.threads)
        threads, version = args.threads or "default", f"onnxruntime {ort.__version__}"

    cfg = load_config(resolve(args.config))
    texts = [r["text"] for r in load_split(resolve(cfg["data"]["splits_dir"]), "val")]
    for i in range(args.warmup):
        clf.predict(texts[i % len(texts)])
    times_ms = []
    for i in range(args.runs):
        start = time.perf_counter()
        clf.predict(texts[i % len(texts)])
        times_ms.append((time.perf_counter() - start) * 1000)

    result = {
        "backend": args.backend,
        "runs": args.runs,
        "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
        "mean_ms": round(float(np.mean(times_ms)), 2),
        "threads": threads,
        "library": version,
        "cpu": platform.processor(),
        "platform": platform.platform(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
