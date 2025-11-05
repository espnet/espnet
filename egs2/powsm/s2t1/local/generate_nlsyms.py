from utils import (
    LANGUAGES,
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

    with open("local/panphon_ipas", "r") as f:
        # read lines and strip whitespace
        panphon_ipas = f.readlines()
        panphon_ipas = sorted([line.strip() for line in panphon_ipas])
    PHONEME_VOCABULARY = set(panphon_ipas)

    with open("data/bpe_nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
        for phoneme in sorted(list(PHONEME_VOCABULARY)):
            fp.write(f"/{phoneme}/\n")

    with open("data/nlsyms.txt", "w") as fp:
        # tokens in nlsyms.txt will be removed during scoring
        # we do not want phonemes to be removed though
        # we only want to remove the / for evaluation
        for tok in special_tokens:
            fp.write(f"{tok}\n")
        fp.write("/\n")
