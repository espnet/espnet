import argparse
import logging
import os
import traceback
from pathlib import Path

try:
    from iso639 import Language
except Exception:
    raise ImportError(
        "iso639 is not installed. Please install it with `pip install python-iso639`"
    )

try:
    from datasets import load_dataset
except Exception:
    raise ImportError(
        "datasets is not installed. Please install it with `pip install datasets`"
    )

FLEURS_LANG_ID = [
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
]
FLEURS_LANG_ID_INDEX = {
    0: "af_za",
    1: "am_et",
    2: "ar_eg",
    3: "as_in",
    4: "ast_es",
    5: "az_az",
    6: "be_by",
    7: "bg_bg",
    8: "bn_in",
    9: "bs_ba",
    10: "ca_es",
    11: "ceb_ph",
    12: "ckb_iq",
    13: "cmn_hans_cn",
    14: "cs_cz",
    15: "cy_gb",
    16: "da_dk",
    17: "de_de",
    18: "el_gr",
    19: "en_us",
    20: "es_419",
    21: "et_ee",
    22: "fa_ir",
    23: "ff_sn",
    24: "fi_fi",
    25: "fil_ph",
    26: "fr_fr",
    27: "ga_ie",
    28: "gl_es",
    29: "gu_in",
    30: "ha_ng",
    31: "he_il",
    32: "hi_in",
    33: "hr_hr",
    34: "hu_hu",
    35: "hy_am",
    36: "id_id",
    37: "ig_ng",
    38: "is_is",
    39: "it_it",
    40: "ja_jp",
    41: "jv_id",
    42: "ka_ge",
    43: "kam_ke",
    44: "kea_cv",
    45: "kk_kz",
    46: "km_kh",
    47: "kn_in",
    48: "ko_kr",
    49: "ky_kg",
    50: "lb_lu",
    51: "lg_ug",
    52: "ln_cd",
    53: "lo_la",
    54: "lt_lt",
    55: "luo_ke",
    56: "lv_lv",
    57: "mi_nz",
    58: "mk_mk",
    59: "ml_in",
    60: "mn_mn",
    61: "mr_in",
    62: "ms_my",
    63: "mt_mt",
    64: "my_mm",
    65: "nb_no",
    66: "ne_np",
    67: "nl_nl",
    68: "nso_za",
    69: "ny_mw",
    70: "oc_fr",
    71: "om_et",
    72: "or_in",
    73: "pa_in",
    74: "pl_pl",
    75: "ps_af",
    76: "pt_br",
    77: "ro_ro",
    78: "ru_ru",
    79: "sd_in",
    80: "sk_sk",
    81: "sl_si",
    82: "sn_zw",
    83: "so_so",
    84: "sr_rs",
    85: "sv_se",
    86: "sw_ke",
    87: "ta_in",
    88: "te_in",
    89: "tg_tj",
    90: "th_th",
    91: "tr_tr",
    92: "uk_ua",
    93: "umb_ao",
    94: "ur_pk",
    95: "uz_uz",
    96: "vi_vn",
    97: "wo_sn",
    98: "xh_za",
    99: "yo_ng",
    100: "yue_hant_hk",
    101: "zu_za",
}


def preprocess_text(text: str) -> str:
    # Note(jinchuan): not sure how should we treat a text
    # that is naturally with "<" or ">"
    if "<" in text or ">" in text:
        logging.warning(f"find an invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text


def collect_data(examples, lang_id_dict):
    ans = []
    for idx, eg in enumerate(examples):

        if not Path(eg["audio"]["path"]).is_file():
            logging.warning(
                f"skip the example due to missing file: {eg['audio']['path']}"
            )
            continue

        lang_id = FLEURS_LANG_ID_INDEX.get(eg["lang_id"], None)
        if lang_id is None:
            logging.warning(f"skip the example due to missing lang_id: {eg['lang_id']}")
            continue

        speech_id = eg["path"].split("/")[-1].replace(".wav", "")
        iso3 = lang_id_dict[lang_id]

        ans.append(
            {
                "utt_id": f"{iso3}_{speech_id}",
                "wav_path": eg["audio"]["path"],
                "iso3": iso3,
            }
        )

        if idx > 0 and idx % 1000 == 0:
            logging.info(f"processed {idx} examples")
    return ans


def main():
    parser = argparse.ArgumentParser(description="Download and format FLEURS dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory for prepared datasets",
    )

    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    fleurs_asr = load_dataset(
        "google/xtreme_s", f"fleurs.all", num_proc=16, trust_remote_code=True
    )

    # Language IDs are in ISO-639-3 format.
    # Some of the results are not exactly identical to:
    # https://arxiv.org/pdf/2205.12446.pdf since we only
    # search the ISO code by the name.
    lang_id_dict = {}

    for lang_id in list(set(FLEURS_LANG_ID)):
        lang_id_short = lang_id.split("_")[0]
        if len(lang_id_short) == 2:
            lang = Language.from_part1(lang_id_short)
        else:
            lang = Language.from_part3(lang_id_short)
        lang_id_iso = lang.part3
        lang_id_dict[lang_id] = lang_id_iso

    # logging.info("Begin to process dev")
    # dev_fleurs_lang_data = collect_data(fleurs_asr["validation"], lang_id_dict)
    logging.info("Begin to process test")
    test_fleurs_lang_data = collect_data(fleurs_asr["test"], lang_id_dict)
    logging.info("Begin to process train")
    train_fleurs_lang_data = collect_data(fleurs_asr["train"], lang_id_dict)

    utt_splits = {
        # "dev_fleurs_lang": dev_fleurs_lang_data,
        "test_fleurs_lang": test_fleurs_lang_data,
        "train_fleurs_lang": train_fleurs_lang_data,
    }

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split, utts in utt_splits.items():
        write_dir = os.path.join(args.output_dir, split)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        wavscp_fp = open(os.path.join(write_dir, "wav.scp"), "w")  # wav-id wav-path
        utt2lang_fp = open(os.path.join(write_dir, "utt2lang"), "w")

        wavscp = []
        utt2lang = []

        for utt in utts:
            utt_id = utt["utt_id"]
            wav_path = utt["wav_path"]
            iso3 = utt["iso3"]
            wavscp.append(f"{utt_id} {wav_path}\n")
            utt2lang.append(f"{utt_id} {iso3}\n")

        wavscp_fp.writelines(sorted(wavscp))
        utt2lang_fp.writelines(sorted(utt2lang))
        wavscp_fp.close()
        utt2lang_fp.close()
        logging.info(f"Finished writing {split} to {write_dir}")


if __name__ == "__main__":
    main()
