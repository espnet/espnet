import argparse


def gen_wav_scp():
    dev_wav_scp = "data/dev_voxlingua107/wav.scp"
    train_wav_scp = "data/train_voxlingua107/wav.scp"
    lang2_to_iso3 = convert_voxlingua107_lang()

    dev_wav_scp_dump = []
    dev_utt_ids = set()
    with open(dev_wav_scp, "r") as f:
        for line in f:
            utt_id = line.split()[0]
            wav_dir = line.split()[1]
            lang = lang2_to_iso3[wav_dir.split("/")[-2]]
            dev_utt_ids.add(utt_id)
            utt_id_lang = f"{lang}_{utt_id}"
            line_lang = f"{utt_id_lang} {wav_dir}\n"
            dev_wav_scp_dump.append(line_lang)

    train_wav_scp_clear = []
    with open(train_wav_scp, "r") as f:
        for line in f:
            utt_id = line.split()[0]
            wav_dir = line.split()[1]
            lang = lang2_to_iso3[wav_dir.split("/")[-2]]
            if utt_id in dev_utt_ids:
                continue
            utt_id_lang = f"{lang}_{utt_id}"
            line_lang = f"{utt_id_lang} {wav_dir}\n"
            train_wav_scp_clear.append(line_lang)

    with open(train_wav_scp, "w") as f:
        f.writelines(sorted(train_wav_scp_clear))
    with open(dev_wav_scp, "w") as f:
        f.writelines(sorted(dev_wav_scp_dump))


def gen_utt2lang():
    train_wav_scp = "data/train_voxlingua107/wav.scp"
    dev_wav_scp = "data/dev_voxlingua107/wav.scp"
    lang2_to_iso3 = convert_voxlingua107_lang()

    train_utt2lang_dump = []
    dev_utt2lang_dump = []
    with open(train_wav_scp, "r") as f:
        for line in f:
            utt_id = line.split()[0]
            wav_dir = line.split()[1]
            lang = lang2_to_iso3[wav_dir.split("/")[-2]]
            train_utt2lang_dump.append(f"{utt_id} {lang}\n")

    with open(dev_wav_scp, "r") as f:
        for line in f:
            utt_id = line.split()[0]
            wav_dir = line.split()[1]
            lang = lang2_to_iso3[wav_dir.split("/")[-2]]
            dev_utt2lang_dump.append(f"{utt_id} {lang}\n")

    with open("data/train_voxlingua107/utt2lang", "w") as f:
        f.writelines(sorted(train_utt2lang_dump))

    with open("data/dev_voxlingua107/utt2lang", "w") as f:
        f.writelines(sorted(dev_utt2lang_dump))


def convert_voxlingua107_lang():
    try:
        import pycountry
    except ImportError:
        raise ImportError("Please install pycountry: pip install pycountry")

    lang2_to_language = {
        "ab": "Abkhazian",
        "af": "Afrikaans",
        "am": "Amharic",
        "ar": "Arabic",
        "as": "Assamese",
        "az": "Azerbaijani",
        "ba": "Bashkir",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "bo": "Tibetan",
        "br": "Breton",
        "bs": "Bosnian",
        "ca": "Catalan",
        "ceb": "Cebuano",
        "cs": "Czech",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "el": "Modern Greek (1453-)",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "fo": "Faroese",
        "fr": "French",
        "gl": "Galician",
        "gn": "Guarani",
        "gu": "Gujarati",
        "gv": "Manx",
        "ha": "Hausa",
        "haw": "Hawaiian",
        "hi": "Hindi",
        "hr": "Croatian",
        "ht": "Haitian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "ia": "Interlingua (International Auxiliary Language Association)",
        "id": "Indonesian",
        "is": "Icelandic",
        "it": "Italian",
        "iw": "Hebrew",  # voxlingua107 uses iw, but the modern one is he
        "ja": "Japanese",
        "jw": "Javanese",  # voxlingua107 uses jw, but the modern one is jv
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "la": "Latin",
        "lb": "Luxembourgish",
        "ln": "Lingala",
        "lo": "Lao",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mg": "Malagasy",
        "mi": "Maori",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mr": "Marathi",
        "ms": "Malay (macrolanguage)",
        "mt": "Maltese",
        "my": "Burmese",
        "ne": "Nepali (macrolanguage)",
        "nl": "Dutch",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "oc": "Occitan (post 1500)",
        "pa": "Panjabi",
        "pl": "Polish",
        "ps": "Pushto",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sa": "Sanskrit",
        "sco": "Scots",
        "sd": "Sindhi",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "sn": "Shona",
        "so": "Somali",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili (macrolanguage)",
        "ta": "Tamil",
        "te": "Telugu",
        "tg": "Tajik",
        "th": "Thai",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tr": "Turkish",
        "tt": "Tatar",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "vi": "Vietnamese",
        "war": "Waray (Philippines)",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "zh": "Mandarin Chinese",
    }

    lang2_to_iso3 = {}

    for lang2, language in lang2_to_language.items():
        lang_data = pycountry.languages.lookup(language)
        iso3_code = lang_data.alpha_3
        lang2_to_iso3[lang2] = iso3_code

    return lang2_to_iso3


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare voxlingua107 dataset")
    parser.add_argument(
        "--func_name",
        type=str,
        help="The function name to run",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    globals()[args.func_name]()
