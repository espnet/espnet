"""OWSM utterance layer shared by every sub-dataset.

Port of ``egs2/owsm_v1/s2t1/local/utils.py``. The merge/packing functions are
copied rather than rewritten so that output stays byte-comparable with the
espnet2 dumps; do not refactor them.

One thing the v1 script does not cover: OWSM v3 onwards rewrites the language
tag in the *text* stream to ISO 639-3 (``egs2/owsm_v3/s2t1/local/filter_lang_id.py``)
while the utterance id keeps the two-letter code minted earlier by
:func:`merge_short_utterances`. A v4 row therefore reads::

    MuST-C_v1.2_ted_767_000012750_000041270_en_asr   <eng><asr><0.00> I'm going...

:func:`lang_token` and :func:`task_token` produce the text-side spelling;
utterance ids must keep using the corpus's own code.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SYMBOL_NA: str = "<na>"
SYMBOL_NOSPEECH: str = "<nospeech>"
SPEECH_MAX_LEN: float = 30
SPEECH_RESOLUTION: float = 0.02

SYMBOLS_TIME: List[str] = [
    "<notimestamps>",
    *[
        f"<{i * SPEECH_RESOLUTION:.2f}>"
        for i in range(round(SPEECH_MAX_LEN / SPEECH_RESOLUTION) + 1)
    ],
]

# Copied from OpenAI Whisper's tokenizer.py, via egs2/owsm_v1/s2t1/local/utils.py.
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# Frozen rather than derived. Upstream builds this from the `iso639` package
# (egs2/owsm_v3/s2t1/local/utils.py:143-153), which needs a git clone, and its
# final `TO_ISO_LANGUAGE_CODE[k] = lang.part3` runs unconditionally -- so the
# `jw` and `my` special cases are overwritten by whatever `lang` the previous
# iteration left behind. The values here are the intended ones.
#
# 89 of these are confirmed against two independent OWSM releases: the local
# owsm_data_v3.2/nlsyms.txt and the published espnet/owsm_v3 bpe.model. Both
# carry the expanded language inventory, and both agree exactly on what is
# missing, so the remaining nine (ms la sq yi fo sa bo mg haw) are languages no
# released OWSM emits rather than mappings in doubt. `no` is the one the oracles
# actually settle: both ship <nob> and <nno> but not the macrolanguage <nor>,
# so Norwegian is Bokmal here.
TO_ISO_LANGUAGE_CODE = {
    "en": "eng",
    "zh": "zho",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "ko": "kor",
    "fr": "fra",
    "ja": "jpn",
    "pt": "por",
    "tr": "tur",
    "pl": "pol",
    "ca": "cat",
    "nl": "nld",
    "ar": "ara",
    "sv": "swe",
    "it": "ita",
    "id": "ind",
    "hi": "hin",
    "fi": "fin",
    "vi": "vie",
    "he": "heb",
    "uk": "ukr",
    "el": "ell",
    "ms": "msa",
    "cs": "ces",
    "ro": "ron",
    "da": "dan",
    "hu": "hun",
    "ta": "tam",
    "no": "nob",
    "th": "tha",
    "ur": "urd",
    "hr": "hrv",
    "bg": "bul",
    "lt": "lit",
    "la": "lat",
    "mi": "mri",
    "ml": "mal",
    "cy": "cym",
    "sk": "slk",
    "te": "tel",
    "fa": "fas",
    "lv": "lav",
    "bn": "ben",
    "sr": "srp",
    "az": "aze",
    "sl": "slv",
    "kn": "kan",
    "et": "est",
    "mk": "mkd",
    "br": "bre",
    "eu": "eus",
    "is": "isl",
    "hy": "hye",
    "ne": "nep",
    "mn": "mon",
    "bs": "bos",
    "kk": "kaz",
    "sq": "sqi",
    "sw": "swa",
    "gl": "glg",
    "mr": "mar",
    "pa": "pan",
    "si": "sin",
    "km": "khm",
    "sn": "sna",
    "yo": "yor",
    "so": "som",
    "af": "afr",
    "oc": "oci",
    "ka": "kat",
    "be": "bel",
    "tg": "tgk",
    "sd": "snd",
    "gu": "guj",
    "am": "amh",
    "yi": "yid",
    "lo": "lao",
    "uz": "uzb",
    "fo": "fao",
    "ht": "hat",
    "ps": "pus",
    "tk": "tuk",
    "nn": "nno",
    "mt": "mlt",
    "sa": "san",
    "lb": "ltz",
    "my": "mya",
    "bo": "bod",
    "tl": "tgl",
    "mg": "mlg",
    "as": "asm",
    "tt": "tat",
    "haw": "haw",
    "ln": "lin",
    "ha": "hau",
    "ba": "bak",
    "jw": "jav",
    "su": "sun",
}


def iso3(code: str) -> str:
    """Return the ISO 639-3 code the text stream uses for ``code``.

    An ISO 639-3 code passes straight through, so a sub-dataset whose corpus
    labels data in ISO form needs no entry in the two-letter table -- which is
    what keeps a new corpus confined to its own directory.
    """
    if code in _ISO_CODES:
        return code
    try:
        return TO_ISO_LANGUAGE_CODE[code]
    except KeyError:
        raise ValueError(f"No ISO 639-3 code for language {code!r}") from None


def lang_token(code: str) -> str:
    """``"en"`` -> ``"<eng>"``, the spelling the v4 text stream uses."""
    return f"<{iso3(code)}>"


def task_token(task: str, target: Optional[str] = None) -> str:
    """``("asr", None)`` -> ``"<asr>"``; ``("st", "de")`` -> ``"<st_deu>"``."""
    if task == "asr":
        return "<asr>"
    if task != "st":
        raise ValueError(f"Unknown task {task!r}; expected 'asr' or 'st'")
    if target is None:
        raise ValueError("task='st' requires a target language")
    return f"<st_{iso3(target)}>"


# The languages the published owsm_v4_medium_1B model carries, in its own order
# (alphabetical by ISO 639-3). Frozen rather than derived: the token ids a model
# trains with depend on this order, and no script in the repo emits it -- v1's
# generate_nlsyms.py spells languages with two-letter Whisper keys instead.
LANGUAGES_ISO: List[str] = [
    "abk",
    "afr",
    "amh",
    "ara",
    "asm",
    "ast",
    "aze",
    "bak",
    "bas",
    "bel",
    "ben",
    "bos",
    "bre",
    "bul",
    "cat",
    "ceb",
    "ces",
    "chv",
    "ckb",
    "cmn",
    "cnh",
    "cym",
    "dan",
    "deu",
    "dgd",
    "div",
    "ell",
    "eng",
    "epo",
    "est",
    "eus",
    "fas",
    "fil",
    "fin",
    "fra",
    "frr",
    "ful",
    "gle",
    "glg",
    "grn",
    "guj",
    "hat",
    "hau",
    "heb",
    "hin",
    "hrv",
    "hsb",
    "hun",
    "hye",
    "ibo",
    "ina",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kab",
    "kam",
    "kan",
    "kat",
    "kaz",
    "kea",
    "khm",
    "kin",
    "kir",
    "kmr",
    "kor",
    "lao",
    "lav",
    "lga",
    "lin",
    "lit",
    "ltz",
    "lug",
    "luo",
    "mal",
    "mar",
    "mas",
    "mdf",
    "mhr",
    "mkd",
    "mlt",
    "mon",
    "mri",
    "mrj",
    "mya",
    "myv",
    "nan",
    "nep",
    "nld",
    "nno",
    "nob",
    "npi",
    "nso",
    "nya",
    "oci",
    "ori",
    "orm",
    "ory",
    "pan",
    "pol",
    "por",
    "pus",
    "quy",
    "roh",
    "ron",
    "rus",
    "sah",
    "sat",
    "sin",
    "skr",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "sot",
    "spa",
    "srd",
    "srp",
    "sun",
    "swa",
    "swe",
    "swh",
    "tam",
    "tat",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tig",
    "tir",
    "tok",
    "tpi",
    "tsn",
    "tuk",
    "tur",
    "twi",
    "uig",
    "ukr",
    "umb",
    "urd",
    "uzb",
    "vie",
    "vot",
    "wol",
    "xho",
    "yor",
    "yue",
    "zho",
    "zul",
]

# The translation directions v4 defines. Far fewer than the languages, because
# most are source-only. A corpus offering a target outside this list cannot be
# tokenized by a v4 vocabulary.
ST_TARGETS_ISO: List[str] = [
    "ara",
    "cat",
    "ces",
    "cym",
    "deu",
    "eng",
    "est",
    "fas",
    "fra",
    "ind",
    "ita",
    "jpn",
    "lav",
    "mon",
    "nld",
    "por",
    "ron",
    "rus",
    "slv",
    "spa",
    "swe",
    "tam",
    "tur",
    "vie",
    "zho",
]


#: Lets iso3() pass an ISO 639-3 code straight through.
_ISO_CODES = frozenset(LANGUAGES_ISO)


def nlsyms() -> List[str]:
    """Return the OWSM special symbols, in the order the vocabulary wants them.

    Reproduces the published owsm_v4_medium_1B inventory exactly -- 1681
    symbols: ``<na>``, ``<nospeech>``, 151 languages, ``<asr>``, 25 translation
    directions, and 1502 timestamps. A test asserts set equality and ordering
    against that model, so this cannot drift from it silently.

    Giving the full inventory rather than only what the wired-up corpora emit is
    deliberate: adding a corpus should mean adding a directory under
    ``sub_datasets/`` and nothing else. A vocabulary sized to today's corpora
    would have to be rebuilt, and every model retrained, the first time a corpus
    arrived with a language it had never seen.
    """
    return [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"<{code}>" for code in LANGUAGES_ISO],
        "<asr>",
        *[f"<st_{code}>" for code in ST_TARGETS_ISO],
        *SYMBOLS_TIME,
    ]


@dataclass
class Utterance:
    """One segment as the egs2 prep scripts represent it, before packing."""

    utt_id: str
    wav_id: str
    wav_path: str
    start_time: float  # in seconds
    end_time: float  # in seconds
    lang: str  # language token of speech
    task: str  # task token
    text: str  # target text without timestamps
    asr_text: str  # source text for CTC ASR without timestamps


@dataclass
class LongUtterance(Utterance):
    """A packed span of up to 30 s, conditioned on the span before it."""

    prev_text: str  # previous (target) text as condition
    text_with_time: str  # target text with timestamps


def time2token(x: float) -> str:
    """Convert float time to timestamp token."""
    x = round(x / SPEECH_RESOLUTION) * SPEECH_RESOLUTION
    return f"<{x:.2f}>"


def merge_short_utterances(
    utts: List[Utterance], prev: Optional[LongUtterance] = None
) -> LongUtterance:
    """Merge a list of utterances to create a long utterance."""
    wav_id = utts[0].wav_id
    wav_path = utts[0].wav_path
    start_time = utts[0].start_time
    end_time = utts[-1].end_time
    lang = utts[0].lang
    task = utts[0].task
    utt_id = (
        f"{wav_id}_{round(1000 * start_time):09d}_"
        f"{round(1000 * end_time):09d}_{lang[1:-1]}_{task[1:-1]}"
    )
    text = " ".join([u.text for u in utts])
    asr_text = " ".join([u.asr_text for u in utts])
    prev_text = prev.text if prev is not None else SYMBOL_NA

    text_with_time = ""
    for u in utts:
        text_with_time += (
            f"{time2token(u.start_time - start_time)} "
            f"{u.text.strip()}{time2token(u.end_time - start_time)}"
        )

    return LongUtterance(
        utt_id=utt_id,
        wav_id=wav_id,
        wav_path=wav_path,
        start_time=start_time,
        end_time=end_time,
        lang=lang,
        task=task,
        text=text,
        asr_text=asr_text,
        prev_text=prev_text,
        text_with_time=text_with_time,
    )


def generate_long_utterances(
    utts: List[Utterance],
) -> List[LongUtterance]:
    """Generate a list of long utterances from a list of short utterances."""
    utts.sort(key=lambda x: x.start_time)

    long_utts = [None]
    left, right = 0, 0
    while left < len(utts):
        if right < len(utts) and (
            utts[right].end_time - utts[left].start_time <= SPEECH_MAX_LEN
        ):
            right += 1
        elif right > left:
            long_utts.append(merge_short_utterances(utts[left:right], long_utts[-1]))
            left = right
        else:
            # An utterance longer than the limit is dropped, and the None keeps
            # the next span's prev_text at <na> instead of reaching past it.
            long_utts.append(None)
            left = right + 1
            right = left

    long_utts = [u for u in long_utts if u is not None]
    return long_utts


CACHE_COLUMNS = (
    "utt_id",
    "wav_path",
    "start_time",
    "end_time",
    "lang",
    "task",
    "tgt_lang",
    "text",
    "text_prev",
    "text_ctc",
)


def check_cache_row(row: dict) -> dict:
    """Return ``row`` unchanged if it carries exactly ``CACHE_COLUMNS``.

    Every sub-dataset writes this one schema, because ``CombinedDataset``
    refuses a mixture whose entries yield different sample keys and a cache
    built with the wrong columns would only fail there -- after the build.
    """
    missing = sorted(set(CACHE_COLUMNS) - set(row))
    unexpected = sorted(set(row) - set(CACHE_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "cache row does not match the shared OWSM schema: "
            f"missing {missing}, unexpected {unexpected}"
        )
    return row


def run_parallel(function, tasks, default_workers: int = 16):
    """Map ``function`` over ``tasks``, fanning out when a cluster is configured.

    It will honour ``espnet3.parallel`` when a recipe has set one, and degrade
    rather than fail when it has not. ``function`` must be module-level and take
    only its task, because on a cluster it is pickled to the worker.
    """
    tasks = list(tasks)
    if not tasks:
        return []

    # Deferred: espnet3.parallel.parallel builds CLUSTER_MAP at module scope, so
    # importing it pulls in dask. The submodule, not the package, which
    # re-exports nothing.
    from espnet3.parallel.parallel import get_client, get_parallel_config

    config = get_parallel_config()
    env = getattr(config, "env", "local") if config is not None else "local"

    if env != "local":
        logger.info("running %d tasks on the %s cluster", len(tasks), env)
        with get_client(config) as client:
            return list(client.gather(client.map(function, tasks)))

    # espnet3 reads `env: local` as "no Dask cluster" and runs in-process, so
    # building a LocalCluster here would break that convention.
    if config is not None:
        workers = int(getattr(config, "n_workers", 1) or 1)
    else:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        workers = min(default_workers, 4 * slurm_cpus)

    if workers <= 1:
        logger.info("running %d tasks serially", len(tasks))
        return [function(task) for task in tasks]

    logger.info("running %d tasks on %d local threads", len(tasks), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, tasks))


def cache_root(recipe_dir, cache: dict | None, corpus: str) -> Path:
    """Return the ``hf_audio_index`` directory holding ``corpus``' splits."""
    root = Path((cache or {}).get("cache_dir", "data/hf"))
    if not root.is_absolute():
        root = Path(recipe_dir) / root
    return root / corpus / "hf_audio_index"


def sub_dataset_config(module: str):
    """Return the ``config.yaml`` sitting beside ``module``, or None.

    ``module`` is a dotted name such as
    ``egs3.owsm_v4.owsm.dataset.sub_datasets.spgispeech.builder``; the config is
    read from its package. Returns None when there is none, so an intermediate
    base class can be declared without one.
    """
    from importlib import resources

    from espnet3.utils.config_utils import load_config_with_defaults

    package = module.rsplit(".", 1)[0]
    try:
        resource = resources.files(package).joinpath("config.yaml")
    except Exception:  # noqa: BLE001
        # A subclass defined in __main__, a notebook or a test has no package
        # to look beside. That is not an error: it just has no config.
        return None
    if not resource.is_file():
        return None
    with resources.as_file(resource) as path:
        return load_config_with_defaults(str(path), resolve=False)
