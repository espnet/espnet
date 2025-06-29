from utils import (
    LANGUAGES,
    PHONEME_VOCABULARY,
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    TASK_TOKENS,
)

# source: https://github.com/espnet/espnet/blob/master/
#         egs2/owsm_v1/s2t1/local/generate_nlsyms.py
if __name__ == "__main__":
    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"{s}" for s in LANGUAGES],
        *TASK_TOKENS,
        *SYMBOLS_TIME,
    ]

    with open("data/bpe_nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
        for phoneme in PHONEME_VOCABULARY:
            fp.write(f"/{phoneme}/\n")

    with open("data/nlsyms.txt", "w") as fp:
        # tokens in nlsyms.txt will be removed during scoring
        # we do not want phonemes to be removed though
        # we only want to remove the / for evaluation
        for tok in special_tokens:
            fp.write(f"{tok}\n")
        fp.write("/\n")
