from typing import Dict, List, Optional, Tuple, Union

# adapted from https://github.com/espnet/espnet/blob/master/egs2/
#              owsm_v1/s2t1/local/utils.py
############################################
# Definitions of shared symbols and values #
############################################
SYMBOL_NA: str = "<na>"  # symbol denoting text is not available
SYMBOL_NOSPEECH: str = "<nospeech>"  # symbol denoting non-speech audio
SPEECH_MAX_LEN: float = 20  # max speech length in seconds
SPEECH_RESOLUTION: float = 0.02  # resolution in seconds
# all timestamp symbols
SYMBOLS_TIME: List[str] = [
    "<notimestamps>",
    *[
        f"<{i * SPEECH_RESOLUTION:.2f}>"
        for i in range(round(SPEECH_MAX_LEN / SPEECH_RESOLUTION) + 1)
    ],
]
TASK_TOKENS: List[str] = ["<asr>", "<pr>", "<g2p>", "<p2g>"]
LANGUAGES = {
    "<afr>": "Afrikaans",
    "<amh>": "Amharic",
    "<ara>": "Arabic",
    "<aze>": "Azerbaijani",
    "<bak>": "Bashkir",
    "<bel>": "Belarusian",
    "<ben>": "Bangla",
    "<bos>": "Bosnian",
    "<bul>": "Bulgarian",
    "<cat>": "Catalan",
    "<ceb>": "Cebuano",
    "<ces>": "Czech",
    "<cmn>": "Mandarin Chinese",
    "<cym>": "Welsh",
    "<dan>": "Danish",
    "<deu>": "German",
    "<ell>": "Greek",
    "<eng>": "English",
    "<est>": "Estonian",
    "<eus>": "Basque",
    "<fin>": "Finnish",
    "<fra>": "French",
    "<ful>": "Fula",
    "<gle>": "Irish",
    "<glg>": "Galician",
    "<hau>": "Hausa",
    "<hin>": "Hindi",
    "<hrv>": "Croatian",
    "<hun>": "Hungarian",
    "<ina>": "Interlingua",
    "<ind>": "Indonesian",
    "<isl>": "Icelandic",
    "<ita>": "Italian",
    "<jav>": "Javanese",
    "<jpn>": "Japanese",
    "<kat>": "Georgian",
    "<kaz>": "Kazakh",
    "<kin>": "Kinyarwanda",
    "<kir>": "Kyrgyz",
    "<kmr>": "Northern Kurdish",
    "<kor>": "Korean",
    "<lao>": "Lao",
    "<lit>": "Lithuanian",
    "<mal>": "Malayalam",
    "<mar>": "Marathi",
    "<mkd>": "Macedonian",
    "<mlt>": "Maltese",
    "<mon>": "Mongolian",
    "<mri>": "Māori",
    "<msa>": "Malay",
    "<mya>": "Burmese",
    "<nld>": "Dutch",
    "<nob>": "Norwegian Bokmål",
    "<nya>": "Nyanja",
    "<ori>": "Odia",
    "<orm>": "Oromo",
    "<pan>": "Punjabi",
    "<pol>": "Polish",
    "<por>": "Portuguese",
    "<ron>": "Romanian",
    "<rus>": "Russian",
    "<sin>": "Sinhala",
    "<skr>": "Saraiki",
    "<slk>": "Slovak",
    "<slv>": "Slovenian",
    "<sna>": "Shona",
    "<snd>": "Sindhi",
    "<som>": "Somali",
    "<spa>": "Spanish",
    "<srp>": "Serbian",
    "<swa>": "Swahili",
    "<swe>": "Swedish",
    "<tam>": "Tamil",
    "<tat>": "Tatar",
    "<tel>": "Telugu",
    "<tgk>": "Tajik",
    "<tha>": "Thai",
    "<tur>": "Turkish",
    "<uig>": "Uyghur",
    "<ukr>": "Ukrainian",
    "<urd>": "Urdu",
    "<uzb>": "Uzbek",
    "<vie>": "Vietnamese",
    "<xho>": "Xhosa",
    "<yor>": "Yoruba",
    "<yue>": "Cantonese",
    "<zul>": "Zulu",
}

# PHONEME_VOCABULARY defined in local/generate_nlsyms.py
