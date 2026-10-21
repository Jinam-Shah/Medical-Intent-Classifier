"""Rule-based red-flag check that runs before the classifier.

The classifier can be confidently wrong on symptoms it was never trained on (it answered
"I have chest pain" as `headache` at 0.996). Emergency symptoms must never get a routine
self-care reply, so a small, explicit pattern list overrides the model. It is a safety net,
not a diagnosis, and deliberately errs toward false alarms.
"""

import re

EMERGENCY_MESSAGE = (
    "This could be a medical emergency. Please call your local emergency number (911 in the US) "
    "or go to the nearest emergency department now. Don't wait for symptoms to get worse."
)
CRISIS_MESSAGE = (
    "I'm really sorry you're feeling this way. You don't have to go through it alone. In the US, "
    "call or text 988 (Suicide & Crisis Lifeline) any time. Elsewhere, contact your local emergency "
    "number or a crisis line."
)

_EMERGENCY_PATTERNS = [
    r"\bchest (pain|pains|tightness|pressure|hurts?|is tight)\b",
    r"\b(pain|tightness|pressure) in (my|the) chest\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    # ...but not a blocked nose ("can't breathe through my nose" is congestion)
    r"\b(can'?t|cannot|can not|unable to|struggling to) breathe?\b"
    r"(?! (through|out of|from)( (my|the|his|her|your))? nose)",
    r"\b(not|stopped) breathing\b",
    r"\bchoking\b",
    r"\b(passed|passing) out\b|\bfainted\b|\bunconscious\b",
    r"\bseizure\b",
    r"\b(severe|heavy|uncontrollable) bleeding\b|\bbleeding (a lot|heavily|won'?t stop)\b",
    r"\b(face|arm) (is )?(drooping|numb)\b|\bslurred speech\b",
    r"\boverdose\b",
    r"\banaphyla\w*\b|\b(throat|tongue|lips?) (is |are )?swelling\b",
]
_CRISIS_PATTERNS = [
    r"\bsuicid\w*\b",
    r"\bkill (myself|me)\b",
    r"\b(end|take) my (own )?life\b",
    r"\bwant to die\b",
    r"\bself[- ]?harm\b|\bhurt(ing)? myself\b",
]
_EMERGENCY = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)
_CRISIS = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)


def red_flag(text):
    """Return ("crisis" | "emergency", message) if the text matches a red flag, else None."""
    normalized = text.replace("’", "'")
    if _CRISIS.search(normalized):
        return "crisis", CRISIS_MESSAGE
    if _EMERGENCY.search(normalized):
        return "emergency", EMERGENCY_MESSAGE
    return None
