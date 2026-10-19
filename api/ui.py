"""Gradio chat UI, mounted inside the FastAPI app at /ui (enable with ENABLE_UI=1).

It uses the same loaded classifier and threshold as the REST API, so the demo and the
API can't disagree.
"""

import json
import random

import gradio as gr

from src.config import resolve

INTENTS_FILE = "data/intents.json"
EXAMPLES = ["I have had a fever since yesterday", "My throat hurts", "I can't stop coughing",
            "Hello!", "What's the weather like today?"]


def load_responses(path=INTENTS_FILE):
    with open(resolve(path), "r", encoding="utf-8") as f:
        return {it["tag"]: it["responses"] for it in json.load(f)["intents"]}


def make_responder(get_classifier, get_threshold, unsure_message, responses):
    """Return respond(message, history) -> ("", new_history, scores) for the Gradio chat.

    get_classifier and get_threshold are called per message, after the API has loaded the model.
    """

    def respond(message, history):
        message = (message or "").strip()
        if not message:
            return "", history, None
        clf, threshold = get_classifier(), get_threshold()
        probs = clf.predict_proba([message])[0]
        scores = {clf.labels[i]: float(p) for i, p in enumerate(probs)}
        intent, confidence = max(scores.items(), key=lambda kv: kv[1])
        if confidence >= threshold:
            reply = f"{random.choice(responses.get(intent, [unsure_message]))}\n\n*[{intent} | {confidence:.1%}]*"
        else:
            reply = f"{unsure_message}\n\n*[unsure | best guess {intent} at {confidence:.1%}, " \
                    f"threshold {threshold:.0%}]*"
        history = (history or []) + [{"role": "user", "content": message},
                                     {"role": "assistant", "content": reply}]
        return "", history, scores

    return respond


def build_demo(get_classifier, get_threshold, disclaimer, unsure_message, responses):
    respond = make_responder(get_classifier, get_threshold, unsure_message, responses)

    with gr.Blocks(title="Medical Intent Classifier") as demo:
        gr.Markdown(
            "# Medical Intent Classifier\n"
            "Describe a symptom (fever, cold, flu, cough, sore throat, congestion, shortness of breath, "
            "headache) or just say hello. The model predicts one of 12 intents; below a calibrated "
            "confidence threshold it asks you to rephrase.\n\n"
            f"**{disclaimer}**\n\n"
            "[REST API docs](/docs) · [Source code](https://github.com/Jinam-Shah/Medical-Intent-Classifier)"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=380)
                box = gr.Textbox(placeholder="e.g. I have had a fever since yesterday", show_label=False,
                                 max_length=500)
                gr.Examples(EXAMPLES, inputs=box)
            with gr.Column(scale=2):
                scores = gr.Label(label="Top intents (calibrated confidence)", num_top_classes=3)
        box.submit(respond, [box, chatbot], [box, chatbot, scores])
    return demo
