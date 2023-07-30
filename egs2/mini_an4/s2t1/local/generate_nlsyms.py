from typing import List

SYMBOL_NA: str = "<na>"  # symbol denoting text is not available
SYMBOL_NOSPEECH: str = "<nospeech>"  # symbol denoting non-speech audio
SPEECH_MAX_LEN: float = 30  # max speech length in seconds
SPEECH_RESOLUTION: float = 0.02  # resolution in seconds
# all timestamp symbols
SYMBOLS_TIME: List[str] = [
    "<notimestamps>",
    *[
        f"<{i * SPEECH_RESOLUTION:.2f}>"
        for i in range(round(SPEECH_MAX_LEN / SPEECH_RESOLUTION) + 1)
    ],
]
LANGUAGES = {
    "eng": "english",
}


if __name__ == "__main__":
    out = "data/nlsyms.txt"

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"<{s}>" for s in LANGUAGES],
        "<asr>",
        *[f"<st_{t}>" for t in LANGUAGES],
        *SYMBOLS_TIME,
    ]

    with open(out, "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
