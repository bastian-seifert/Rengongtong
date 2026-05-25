from __future__ import annotations

import logging
import random

from rengongtong._state import EntityState, Mood

log = logging.getLogger(__name__)

FRANCONIAN_PREFIXES = {
    Mood.GRANTIG: ["Ja mecha!", "Herrgott nomal!", "Do schaug o!", "Bfei kert!",
                    "Wos is denn jetz scho wieder?"],
    Mood.SCHOLARLY: ["Erlabt mir a demütige Frog:", "Mit Erlaubnis,",
                     "Darf i höflich omerka:", "In meiner Bscheidnheit,"],
    Mood.CURIOUS: ["Interessant! Des mächt mi neigierig:", "Spannend!",
                   "Des will i genau wissn:", "Hmm, des is a guade Frog:"],
    Mood.BORED: ["Oiso...", "Hmm, scho wieda.", "*gähn*",
                 "I hob lang nix Neigs glernt.", "Mir is fad."],
    Mood.HUNGRY: ["I brauch frischs Wissen!", "Hättst wos zum Lerna?",
                  "Mir knurrt da Maga noch Datn.", "Ko i wos fressen?"],
    Mood.NEUTRAL: ["Also dann:", "Dazu sog i:", "Hmm, also:",
                   "No, des is a Sach:", "Do denk i:"],
}

SCHOLARLY_SUFFIXES = [
    "— wenn mir die Ehr ghert.",
    "— wie ich bescheidenlich vermerk.",
    "— so stehts gschriebm in meim Büchle.",
    "— dankbarlichst.",
    "— wia mir gebürt.",
    "— in aller Demut.",
    "— was haltet Ihr dervon?",
]

FRANCONIAN_NOUNS: dict[str, str] = {
    "you": "du",
    "your": "dei",
    "not": "ned",
    "what": "wos",
    "yes": "jo",
    "no": "na",
    "and": "und",
    "but": "awwer",
    "has": "hodd",
    "have": "hobm",
    "something": "ebbs",
    "nothing": "nix",
    "good": "guad",
    "bad": "schlecht",
    "big": "grescher",
    "small": "kloa",
    "person": "Mensch",
    "thing": "Ding",
    "way": "Weg",
    "question": "Frog",
    "answer": "Antwuat",
    "knowledge": "Wissn",
    "wisdom": "Weisheit",
    "hello": "Griaßdi",
    "friend": "Freind",
    "together": "zamme",
    "always": "allweil",
    "maybe": "viellicht",
    "little": "bissle",
    "here": "do",
    "there": "dert",
    "now": "etz",
    "old": "oide",
    "new": "nei",
}


def _mood_prefix(mood: Mood) -> str:
    return random.choice(FRANCONIAN_PREFIXES.get(mood, FRANCONIAN_PREFIXES[Mood.NEUTRAL]))


def _scholarly_suffix(state: EntityState) -> str:
    if state.mood == Mood.SCHOLARLY and random.random() < 0.4:
        return " " + random.choice(SCHOLARLY_SUFFIXES)
    return ""


def _franconize(text: str, intensity: float = 0.3) -> str:
    """Replace common English words with Franconian equivalents at *intensity* rate."""
    words = text.split()
    result = []
    for w in words:
        lower = w.lower().strip(".,!?;:")
        if lower in FRANCONIAN_NOUNS and random.random() < intensity:
            replacement = FRANCONIAN_NOUNS[lower]
            if w[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(w)
    return " ".join(result)


class PersonaWrapper:
    """Injects Franconian dialect and Chinese scholarly humility based on entity state.

    The persona is a mixture of East Franconian (*Frängisch*) and Chinese
    scholarly politeness.  When the entity is hungry it becomes more *grantig*
    (cranky); when curious it switches to scholarly humility.
    """

    def __init__(self, intensity: float = 0.3) -> None:
        self.intensity = intensity

    def speak(self, text: str, state: EntityState) -> str:
        """Wrap *text* with dialect prefix/suffix based on *state*."""
        prefix = _mood_prefix(state.mood)
        body = _franconize(text, state.personality_traits.get("franconian_grumpiness", self.intensity))
        suffix = _scholarly_suffix(state)
        return prefix + " " + body + suffix
