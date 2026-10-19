#!/usr/bin/env python3
"""Interactive command-line chatbot on top of the intent classifier.

    python chatbot.py          # chat
    python chatbot.py test     # run a fixed set of sample inputs

Not medical advice. For emergencies, contact local emergency services.
"""

import json
import random
import sys

from src.config import resolve
from src.predict import DEFAULT_CHECKPOINT, IntentClassifier

INTENTS_PATH = "data/intents.json"
CONFIDENCE_THRESHOLD = 0.75  # provisional; Phase 4 picks this on the validation split
FALLBACK = "I'm not sure I understood. Could you rephrase, or describe your main symptom?"
DISCLAIMER = "This chatbot is a demo and is not medical advice. Consult a healthcare professional."

SAMPLE_INPUTS = [
    "Hello", "Hi there", "I have a fever", "I'm feeling sick", "My throat hurts",
    "I've been coughing", "I feel congested", "I have a headache",
    "How long does the flu last?", "Is this serious?", "Thank you", "Goodbye",
]


class HealthcareChatbot:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, intents_path=INTENTS_PATH):
        self.classifier = IntentClassifier(checkpoint)
        with open(resolve(intents_path), "r", encoding="utf-8") as f:
            self.responses = {it["tag"]: it["responses"] for it in json.load(f)["intents"]}

    def reply(self, text):
        """Return (response, prediction dict)."""
        pred = self.classifier.predict(text)
        if pred["confidence"] < CONFIDENCE_THRESHOLD:
            return FALLBACK, pred
        return random.choice(self.responses.get(pred["intent"], [FALLBACK])), pred

    def chat(self):
        print(DISCLAIMER)
        print("Type 'quit' to exit, 'debug' to toggle prediction details.\n")
        debug = False
        while True:
            try:
                text = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBot: Goodbye! Take care of your health!")
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit"}:
                print("Bot: Goodbye! Take care of your health!")
                break
            if text.lower() == "debug":
                debug = not debug
                print(f"Debug mode {'ON' if debug else 'OFF'}\n")
                continue

            response, pred = self.reply(text)
            print(f"Bot: {response}")
            if debug:
                print("     " + ", ".join(f"{p['intent']}={p['confidence']:.3f}" for p in pred["top_k"]))
            else:
                print(f"     [{pred['intent']} | {pred['confidence']:.1%}]")
            print()


def run_samples(bot):
    for text in SAMPLE_INPUTS:
        response, pred = bot.reply(text)
        print(f"User: {text}\nBot:  {response}\n      [{pred['intent']} | {pred['confidence']:.1%}]\n")


def main():
    try:
        bot = HealthcareChatbot()
    except FileNotFoundError as e:
        print(f"Model not found ({e}). Train it first: python -m src.train")
        sys.exit(1)
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_samples(bot)
    else:
        bot.chat()


if __name__ == "__main__":
    main()
