""" Out-of-orthography statistics for babel predictions."""

from pathlib import Path

langs = ["assamese", "tagalog", "swahili", "zulu", "bengali", "tokpisin",
         "georgian", "haitian", "cantonese", "kurmanji", "turkish", "lao",
         "pashto", "tamil", "tokpisin", "vietnamese"]

# First determine the orthographies from the training data
invs = {}
dicts = Path("dicts")
for lang in langs:
    lang_dict = dicts / "{}_dict.txt".format(lang)
    with lang_dict.open() as f:
        lines = f.readlines()
    inv = set([line.split()[0] for line in lines])
    invs[lang] = inv

alpha = 0.33
beta = 0.33
phoneme_layer = "_phonemelayer2"
# Then gather the predictions from exp dir decodings
#and determine the %age of characters that were in the original orthography
def print_ooo_rate(alpha, beta, phoneme_layer=""):
    print(alpha, beta, phoneme_layer)
    print("Lang,OOO rate")
    for lang in langs:
        total = 0
        ooo = 0
        #hyp_path = """exp/tr_babel10_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.33_phonemeweight0.33_adadelta_bs50_mli800_mlo150_phonemelayer2/decode_et_babel_{}_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/hyp.grapheme.trn""".format(lang)
        hyp_path = """exp/tr_babel10_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha{}_phonemeweight{}_adadelta_bs50_mli800_mlo150{}/decode_et_babel_{}_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/hyp.grapheme.trn""".format(alpha, beta, phoneme_layer, lang)
        with open(hyp_path) as f:
            for line in f:
                toks = line.split()[:-1]
                total += len(toks)
                for tok in toks:
                    if tok not in invs[lang]:
                        ooo += 1
        print("{},{:0.3f}".format(lang, ooo/total))

print_ooo_rate(alpha, beta, phoneme_layer)
