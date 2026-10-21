"""Gradio chat UI, mounted inside the FastAPI app at /ui (enable with ENABLE_UI=1).

It uses the same loaded classifier and threshold as the REST API, so the demo and the
API can't disagree. Red-flag phrases (src/safety.py) are checked before the model.
The symptom list and replies come from the intents file (INTENTS_FILE), so the UI follows the model.
"""

import json
import os
import random

import gradio as gr

from src.config import resolve
from src.safety import red_flag

DEFAULT_INTENTS_FILE = "data/intents.json"
NON_SYMPTOM_INTENTS = {"greeting", "how_are_you", "thanks", "goodbye", "capabilities", "out_of_scope"}
TITLE = "Health Assistant"
EXAMPLES = ["My stomach has been hurting since last night", "I've had a fever since yesterday",
            "My throat is really sore", "I can't sleep at all", "What can you help with?"]
CSS = """
.gradio-container { max-width: 760px !important; margin: 0 auto !important; }
footer { display: none !important; }
"""


def load_responses(path=None):
    path = path or os.getenv("INTENTS_FILE", DEFAULT_INTENTS_FILE)
    with open(resolve(path), "r", encoding="utf-8") as f:
        return {it["tag"]: it["responses"] for it in json.load(f)["intents"]}


def symptom_list(responses):
    names = sorted(tag.replace("_", " ").replace("nausea vomiting", "nausea") for tag in responses
                   if tag not in NON_SYMPTOM_INTENTS)
    return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)


def unsure_reply(responses):
    return ("I'm not sure I understood that. I can help with common symptoms like "
            f"{symptom_list(responses)}. Could you describe what you're feeling in a few words?")


def welcome(responses):
    return ("## 👋 Hi, I'm a symptom chatbot\n"
            f"Tell me how you're feeling. I can share general self-care tips for **{symptom_list(responses)}**.\n\n"
            "*Not medical advice. In an emergency, call your local emergency number.*")


def make_responder(get_classifier, get_threshold, responses, rng=random):
    """Return reply(message, history) -> str for the chat.

    get_classifier and get_threshold are called per message, after the API has loaded the model.
    """
    fallback = unsure_reply(responses)

    def reply(message, history=None):
        message = (message or "").strip()
        if not message:
            return "Please type a message."
        flag = red_flag(message)
        if flag:
            return f"⚠️ **{flag[1]}**"
        clf, threshold = get_classifier(), get_threshold()
        probs = clf.predict_proba([message[:500]])[0]
        best = int(probs.argmax())
        if probs[best] < threshold:
            return fallback
        return rng.choice(responses.get(clf.labels[best]) or [fallback])

    return reply


def build_demo(get_classifier, get_threshold, responses):
    reply = make_responder(get_classifier, get_threshold, responses)
    chatbot = gr.Chatbot(placeholder=welcome(responses), height="68vh", show_label=False, buttons=["copy"])
    textbox = gr.Textbox(placeholder="Describe how you're feeling…", show_label=False, max_length=500,
                         submit_btn=True, autofocus=True)
    return gr.ChatInterface(
        fn=reply,
        chatbot=chatbot,
        textbox=textbox,
        examples=EXAMPLES,
        examples_label="Try",
        title=TITLE,
        flagging_mode="never",
        fill_height=True,
        api_visibility="private",
    )


def mount(app, demo, path="/ui"):
    """Mount without Gradio's footer links (Use via API / Built with Gradio / Settings)."""
    return gr.mount_gradio_app(app, demo, path=path, footer_links=[], theme=gr.themes.Soft(), css=CSS)
