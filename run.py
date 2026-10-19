#!/usr/bin/env python3
"""Shortcut entry point.

    python run.py split    # create the fixed train/val/test split (already committed)
    python run.py train    # train on the training split, select on validation
    python run.py chat     # interactive chatbot
    python run.py test     # chatbot on sample inputs
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Medical Intent Classifier")
    parser.add_argument("command", choices=["split", "train", "chat", "test"])
    parser.add_argument("--config", default="configs/bilstm.yaml")
    args = parser.parse_args()

    if args.command == "split":
        sys.argv = ["src.data", "--config", args.config]
        from src.data import main as split_main
        split_main()
    elif args.command == "train":
        from src.config import load_config, resolve
        from src.train import train
        train(load_config(resolve(args.config)))
    else:
        sys.argv = ["chatbot.py"] + (["test"] if args.command == "test" else [])
        from chatbot import main as chat_main
        chat_main()


if __name__ == "__main__":
    main()
