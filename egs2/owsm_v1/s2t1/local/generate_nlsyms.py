from utils import LANGUAGES, SYMBOL_NA, SYMBOL_NOSPEECH, SYMBOLS_TIME

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
