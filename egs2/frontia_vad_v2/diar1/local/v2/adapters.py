"""All corpus readers, by group, and which of them the training data uses.

Each adapters_<group>.py registers its corpora with registry.corpus(). A
reader gets the corpus directory, the Writers of its splits (prep.py), its
shard, and the keyword arguments in OPTIONS; it adds every recording of its
shard with Writers[split].add().
"""

import adapters_conversation  # noqa: F401
import adapters_farfield_read  # noqa: F401
import adapters_nonspeech  # noqa: F401
import adapters_read_vocal  # noqa: F401
import adapters_synthetic  # noqa: F401
from registry import ADAPTERS, DOMAINS  # noqa: F401

# Registered but left out of the training data (Shinji, 2026-11-09).
NOT_USED = {
    "harpervalleybank": "segment times come from a machine pipeline",
    "dral": "2.6 s speech-only fragments, no boundaries to learn",
    "perit_duplex": "no transcripts or timings",
    "datahive_conv": "no transcripts or timings",
    "pyramic": "5 played sentences in an anechoic room, energy labels",
    "clotho": "unlabelled background speech; voice filter by keywords only",
    "interaction_behavior": "IFADV audio is GPL-2.0 (copyleft, treated as SA)",
    "arabic_speech_corpus": "licence unclear (CC BY 4.0 vs BY-NC-SA mirrors)",
    "children_hri_en": "no consent statement for the children, no transcripts",
    "holoassist": "on hold until the event timing is checked on 3 videos",
    "egomusic": "singing without timed vocal labels",
}

# Keyword arguments of the readers.
OPTIONS = {
    # each of small/medium/large: 509 + 700 + 700 h, about 1,900 h in all
    "libriheavy": dict(max_hours=700),
    "mls": dict(
        max_hours={
            "english": 2000,
            **{
                lang: 300
                for lang in (
                    "dutch",
                    "french",
                    "german",
                    "italian",
                    "polish",
                    "portuguese",
                    "spanish",
                )
            },
        }
    ),
}

# Hours of the train split kept per corpus, and per language in the
# multilingual corpora (dev is not capped). prep.py divides them by the
# number of shards.
MAX_HOURS = 2000
MAX_HOURS_PER_CORPUS = {"babycry": 50}
MAX_HOURS_PER_LANG = {
    corpus: 300
    for corpus in ("commonvoice", "fleurs", "minds14", "commonlanguage", "waxal")
}


def training_corpora():
    """Registered corpora that the training data uses, sorted."""
    return sorted(set(ADAPTERS) - set(NOT_USED))
