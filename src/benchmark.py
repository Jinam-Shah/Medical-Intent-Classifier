"""Measure single-query CPU inference latency (preprocessing + model) on validation sentences.

    python -m src.benchmark --runs 500
"""

import argparse
import json
import platform
import time

import numpy as np
import torch

from src.config import load_config, resolve
from src.data import load_split
from src.predict import DEFAULT_CHECKPOINT, IntentClassifier


def main():
    parser = argparse.ArgumentParser(description="Single-query latency benchmark.")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: torch's)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(resolve(args.config))
    texts = [r["text"] for r in load_split(resolve(cfg["data"]["splits_dir"]), "val")]
    clf = IntentClassifier(args.checkpoint)

    for i in range(args.warmup):
        clf.predict(texts[i % len(texts)])
    times_ms = []
    for i in range(args.runs):
        start = time.perf_counter()
        clf.predict(texts[i % len(texts)])
        times_ms.append((time.perf_counter() - start) * 1000)

    result = {
        "backend": "pytorch",
        "runs": args.runs,
        "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
        "mean_ms": round(float(np.mean(times_ms)), 2),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "cpu": platform.processor(),
        "platform": platform.platform(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
