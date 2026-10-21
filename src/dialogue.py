"""Multi-turn symptom conversation on top of the intent classifier.

The server is stateless: the conversation state is a small JSON object that the client sends
back with every message (`state` in, `state` out). Nothing about a user is stored server-side,
and Render restarts or multiple tabs can't break a conversation.

Flow for a symptom message:
    "my stomach hurts"      -> acknowledge, ask how long            (stage ask_duration)
    "since yesterday"       -> ask how bad                          (stage ask_severity)
    "pretty bad"            -> self-care advice + a line tailored to duration/severity (stage idle)
Messages with several symptoms ("fever and a cough") are split into clauses and handled together.
Red-flag phrases (src/safety.py) are checked on every turn, before anything else.
"""

import json
import os
import re

from src.config import resolve
from src.safety import red_flag

DEFAULT_INTENTS_FILE = "data/intents.json"
CONVERSATIONAL = {"greeting", "how_are_you", "thanks", "goodbye", "capabilities"}
OUT_OF_SCOPE = "out_of_scope"
MAX_SYMPTOMS = 3

DISPLAY = {
    "nausea_vomiting": "nausea", "shortness_of_breath": "shortness of breath", "skin_rash": "a skin rash",
    "body_aches": "body aches", "back_pain": "back pain", "ear_pain": "ear pain", "sore_throat": "a sore throat",
    "stomach_ache": "a stomach ache", "toothache": "a toothache", "headache": "a headache", "cough": "a cough",
    "cold": "a cold", "fever": "a fever", "flu": "the flu", "fatigue": "tiredness", "insomnia": "trouble sleeping",
    "allergy": "allergies", "dizziness": "dizziness", "diarrhea": "diarrhea", "congestion": "congestion",
}
SUGGESTIONS = ["Stomach ache", "Headache", "Sore throat", "Fever", "Can't sleep", "What can you help with?"]
DURATION_CHOICES = ["Started today", "1–3 days", "4–7 days", "Over a week"]
SEVERITY_CHOICES = ["Mild", "Moderate", "Severe"]
SKIP = "Skip"
AFTER_ADVICE = ["I have another symptom", "Thanks!"]

# Free-text answers to the follow-up questions.
_DURATION_PATTERNS = [
    ("over_week", r"\b(over|more than|longer than) (a|one) week\b|\bweeks?\b|\bmonths?\b|\byears?\b|\ba long time\b"
                  r"|\bages\b|\b([89]|\d\d+) days\b|\bover a week\b"),
    ("4_7_days", r"\b[4-7] days\b|\b(four|five|six|seven) days\b|\bfew days\b|\bseveral days\b|\b4[-–]7 days\b"
                 r"|\bsince (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    ("1_3_days", r"\b[1-3] days?\b|\b(a|one|two|three) days?\b|\bcouple (of )?days\b|\byesterday\b"
                 r"|\b1[-–]3 days\b|\blast night\b|\bsince (the )?weekend\b"),
    ("under_1_day", r"\btoday\b|\bthis (morning|afternoon|evening)\b|\b(an|a few|\d+) hours?\b|\bjust (started|now)\b"
                    r"|\btonight\b|\bstarted today\b"),
]
_SEVERITY_PATTERNS = [
    ("mild", r"\bnot (too|that|very|so) bad\b|\bnot (that|too) (painful|serious)\b"),  # before "bad" -> moderate
    ("severe", r"\b(severe|terrible|awful|unbearable|excruciating|worst|really bad|very bad|so bad|pretty bad|"
               r"extreme|intense|horrible|agony|killing me)\b|\b([89]|10) ?(/|out of) ?10\b"),
    ("moderate", r"\b(moderate|medium|bearable|manageable|bad|not great|uncomfortable)\b|\b[4-7] ?(/|out of) ?10\b"),
    ("mild", r"\b(mild|slight|slightly|little|a bit|minor|not (too|that|very) bad|okay|ok|light)\b"
             r"|\b[1-3] ?(/|out of) ?10\b"),
]
_SPLIT = re.compile(r"\b(?:and|also|plus|with|as well as|along with)\b|[,;&+]", re.IGNORECASE)
_SKIP = re.compile(r"^\s*(skip|not sure|don'?t know|idk|no idea|dunno)\s*[.!]?\s*$", re.IGNORECASE)
_ANOTHER = re.compile(r"\banother symptom\b|\bsomething else\b", re.IGNORECASE)


def load_responses(path=None):
    """Intent -> list of reply texts, from INTENTS_FILE (default data/intents.json)."""
    path = path or os.getenv("INTENTS_FILE", DEFAULT_INTENTS_FILE)
    with open(resolve(path), "r", encoding="utf-8") as f:
        return {it["tag"]: it["responses"] for it in json.load(f)["intents"]}


def parse_duration(text):
    t = text.lower()
    for label, pattern in _DURATION_PATTERNS:
        if re.search(pattern, t):
            return label
    return None


def parse_severity(text):
    t = text.lower()
    for label, pattern in _SEVERITY_PATTERNS:
        if re.search(pattern, t):
            return label
    return None


def describe(symptoms):
    names = [DISPLAY.get(s, s.replace("_", " ")) for s in symptoms]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def tailored_note(symptoms, duration, severity):
    """One sentence adjusting the generic advice to how long and how bad."""
    if severity == "severe" and "shortness_of_breath" in symptoms:
        return ("Severe breathlessness needs prompt care: if you're struggling to breathe or your lips look blue, "
                "call emergency services now.")
    if severity == "severe" or duration == "over_week":
        reason = "it's severe" if severity == "severe" else "it has lasted more than a week"
        return f"Because {reason}, it's a good idea to see a doctor soon rather than waiting it out."
    if duration == "4_7_days" or severity == "moderate":
        return "If it isn't clearly improving over the next few days, consider booking a doctor's appointment."
    if duration or severity:
        return "Mild, recent symptoms like this usually get better on their own with self-care."
    return "If it gets worse or doesn't improve within a few days, check in with a doctor or pharmacist."


def clean_state(state):
    """Accept only known keys and sane values from the client-supplied state."""
    state = state if isinstance(state, dict) else {}
    stage = state.get("stage") if state.get("stage") in ("idle", "ask_duration", "ask_severity") else "idle"
    symptoms = [s for s in state.get("symptoms", []) if isinstance(s, str) and s in DISPLAY][:MAX_SYMPTOMS]
    duration = state.get("duration") if state.get("duration") in {p[0] for p in _DURATION_PATTERNS} else None
    severity = state.get("severity") if state.get("severity") in {p[0] for p in _SEVERITY_PATTERNS} else None
    turns = state.get("turns") if isinstance(state.get("turns"), int) and 0 <= state.get("turns") < 10_000 else 0
    if not symptoms:
        stage = "idle"
    return {"stage": stage, "symptoms": symptoms, "duration": duration, "severity": severity, "turns": turns}


class Dialogue:
    def __init__(self, classifier, threshold, responses):
        self.clf = classifier
        self.threshold = threshold
        self.responses = responses

    def _pick(self, intent, turns):
        options = self.responses.get(intent) or []
        return options[turns % len(options)] if options else ""

    def classify(self, text):
        """Return (intent, confidence), with low-confidence predictions mapped to out_of_scope."""
        probs = self.clf.predict_proba([text[:500]])[0]
        best = int(probs.argmax())
        intent, conf = self.clf.labels[best], float(probs[best])
        return (intent if conf >= self.threshold else OUT_OF_SCOPE), conf

    def symptoms_in(self, text):
        """Symptom intents in a message: the whole message, or each clause when there are several."""
        whole, _ = self.classify(text)
        clauses = [c.strip() for c in _SPLIT.split(text) if c and len(c.strip().split()) >= 1 and c.strip()]
        found = []
        if len(clauses) > 1:
            for clause in clauses:
                intent, _ = self.classify(clause)
                if intent in DISPLAY and intent not in found:
                    found.append(intent)
        if len(found) < 2:
            found = [whole] if whole in DISPLAY else found[:1]
        return whole, found[:MAX_SYMPTOMS]

    def _reply(self, text, kind="message", quick_replies=None, intents=None, state=None):
        return {"reply": text, "kind": kind, "quick_replies": quick_replies or [], "intents": intents or [],
                "state": state}

    def _advice(self, state):
        symptoms, turns = state["symptoms"], state["turns"]
        parts = []
        if len(symptoms) > 1:
            parts.append(f"Here's some general guidance for {describe(symptoms)}:")
            parts += [f"**{DISPLAY[s][0].upper() + DISPLAY[s][1:]}:** {self._pick(s, turns)}" for s in symptoms]
        else:
            parts.append(self._pick(symptoms[0], turns))
        parts.append(tailored_note(symptoms, state["duration"], state["severity"]))
        done = {**state, "stage": "idle", "symptoms": [], "duration": None, "severity": None}
        return self._reply("\n\n".join(parts), quick_replies=AFTER_ADVICE, intents=symptoms, state=done)

    def _start_symptoms(self, symptoms, state, duration=None, severity=None):
        state = {**state, "symptoms": symptoms, "duration": duration, "severity": severity}
        lead = f"Sorry to hear about {describe(symptoms)}."
        if duration is None:
            return self._reply(f"{lead} How long have you had {'them' if len(symptoms) > 1 else 'it'}?",
                               quick_replies=DURATION_CHOICES + [SKIP], intents=symptoms,
                               state={**state, "stage": "ask_duration"})
        if severity is None:
            return self._reply(f"{lead} How bad would you say it is?", quick_replies=SEVERITY_CHOICES + [SKIP],
                               intents=symptoms, state={**state, "stage": "ask_severity"})
        return self._advice(state)

    def respond(self, message, state=None):
        state = clean_state(state)
        state["turns"] += 1
        text = (message or "").strip()
        if not text:
            return self._reply("Please type a message.", state=state)

        flag = red_flag(text)
        if flag:
            idle = {**state, "stage": "idle", "symptoms": [], "duration": None, "severity": None}
            return self._reply(flag[1], kind=flag[0], intents=[flag[0]], state=idle)

        # Answering a follow-up question?
        if state["stage"] in ("ask_duration", "ask_severity"):
            if _SKIP.match(text):
                return self._advice(state)
            if state["stage"] == "ask_duration":
                duration = parse_duration(text)
                if duration:
                    severity = parse_severity(text)
                    return self._start_symptoms(state["symptoms"], state, duration, severity) if severity else \
                        self._reply("Thanks. And how bad would you say it is?",
                                    quick_replies=SEVERITY_CHOICES + [SKIP], intents=state["symptoms"],
                                    state={**state, "duration": duration, "stage": "ask_severity"})
            else:
                severity = parse_severity(text)
                if severity:
                    return self._advice({**state, "severity": severity})
            # Not an answer: another symptom ("I also feel dizzy") is added to the ones being discussed;
            # anything else gets the advice we can give now.
            whole, symptoms = self.symptoms_in(text)
            if symptoms:
                merged = (state["symptoms"] + [s for s in symptoms if s not in state["symptoms"]])[:MAX_SYMPTOMS]
                return self._start_symptoms(merged, state, parse_duration(text) or state["duration"],
                                            parse_severity(text) or state["severity"])
            return self._advice(state)

        if _ANOTHER.search(text):
            return self._reply("Sure, what else is bothering you?", quick_replies=SUGGESTIONS[:5], state=state)

        whole, symptoms = self.symptoms_in(text)
        if symptoms:
            return self._start_symptoms(symptoms, state, parse_duration(text), parse_severity(text))
        if whole in CONVERSATIONAL:
            quick = {"greeting": SUGGESTIONS, "how_are_you": SUGGESTIONS[:4], "capabilities": SUGGESTIONS[:5],
                     "thanks": ["I have another symptom"], "goodbye": []}[whole]
            return self._reply(self._pick(whole, state["turns"]), quick_replies=quick, intents=[whole], state=state)
        unsure = ("I'm not sure I can help with that. I can share general self-care tips for common symptoms "
                  "like a stomach ache, headache, sore throat, fever, back pain, or trouble sleeping. "
                  "What are you feeling?")
        return self._reply(unsure, quick_replies=SUGGESTIONS, intents=[OUT_OF_SCOPE], state=state)
