#!/usr/bin/env python3
"""The browser demo of an OWSM model: what `espnet demo` serves.

The first half - the language table, the menus a checkpoint describes, the
reading and padding of audio - is the same ground the two Hugging Face Space
apps cover, at egs2/owsm_ctc_v4/s2t1/demo and egs2/owsm_v4/s2t1/demo. They do
not import it yet, and must not: a Space installs espnet from PyPI, so an
import of a module newer than every release would kill both live demos the
next time one is uploaded. They keep their copies until a release carries
this module; test/espnet2/bin/test_demo_apps.py compares the two against it
and fails when either drifts.

gradio is not a dependency of `pip install espnet` and must not become one: a
Space's requirements.txt and `uvx espnet-mcp` install the bare package to load
a model, and no inference path may need a web framework to import. It is
therefore imported inside the functions that build the app, and
ci/check_inference_imports.py imports this module on a bare install to keep it
that way. `espnet demo` says `pip install espnet[demo]` when it is missing.
"""

import re
from typing import Iterable, List, Sequence, Tuple

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16000
WINDOW_SECS = 30  # what OWSM is trained on; longer audio is decoded in chunks
# A browser demo answers while someone waits, so both Spaces refuse audio
# longer than this and `espnet demo` decodes only the first two minutes of it.
# Neither the model nor `espnet asr` has such a limit.
MAX_SECS = 120
DETECT = "Detect automatically"
ASR_LABEL = "Transcribe"

# ISO 639-3 to English, for the menu. The codes themselves come from the
# loaded model, so a checkpoint covering more languages needs no edit here;
# one without a name falls back to its code.
LANGUAGE_NAMES = {
    "abk": "Abkhazian",
    "afr": "Afrikaans",
    "amh": "Amharic",
    "ara": "Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "aze": "Azerbaijani",
    "bak": "Bashkir",
    "bas": "Basa",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bre": "Breton",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "chv": "Chuvash",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cnh": "Hakha Chin",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "div": "Dhivehi",
    "ell": "Greek",
    "eng": "English",
    "epo": "Esperanto",
    "est": "Estonian",
    "eus": "Basque",
    "fas": "Persian",
    "fil": "Filipino",
    "fin": "Finnish",
    "fra": "French",
    "ful": "Fulah",
    "gle": "Irish",
    "glg": "Galician",
    "grn": "Guarani",
    "guj": "Gujarati",
    "hat": "Haitian",
    "hau": "Hausa",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kab": "Kabyle",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "khm": "Khmer",
    "kin": "Kinyarwanda",
    "kir": "Kirghiz",
    "kor": "Korean",
    "lao": "Lao",
    "lav": "Latvian",
    "lin": "Lingala",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mon": "Mongolian",
    "mri": "Maori",
    "mya": "Burmese",
    "nep": "Nepali",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokmal",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ori": "Oriya",
    "orm": "Oromo",
    "pan": "Panjabi",
    "pol": "Polish",
    "por": "Portuguese",
    "pus": "Pushto",
    "ron": "Romanian",
    "rus": "Russian",
    "sin": "Sinhala",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "sun": "Sundanese",
    "swa": "Swahili",
    "swe": "Swedish",
    "tam": "Tamil",
    "tat": "Tatar",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tir": "Tigrinya",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzb": "Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Yue Chinese",
    "zho": "Chinese",
    "zul": "Zulu",
}


def default_device() -> str:
    """Where to run: the GPU when torch sees one, the CPU otherwise.

    Deliberately not MPS: OWSM-CTC's greedy search is launch-bound there and
    slower than the CPU, so Apple silicon has to be asked for by name.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _named(codes: Iterable[str]) -> List[Tuple[str, str]]:
    """Menu labels for the model's own language codes, sorted by label."""
    return sorted((f"{LANGUAGE_NAMES.get(c, c)} ({c})", c) for c in codes)


def language_codes(tokens: Sequence[str]) -> List[str]:
    """The language symbols a checkpoint carries, whichever OWSM wrote it.

    They sit before <asr> in the token list, three letters each. <unk> looks
    the same and is not one; <sos>, <eos> and <sop> come after <asr>.
    """
    tokens = list(tokens)
    return [
        t[1:-1]
        for t in tokens[: tokens.index("<asr>")]
        if re.fullmatch(r"<[a-z]{3}>", t) and t != "<unk>"
    ]


def target_codes(tokens: Sequence[str]) -> List[str]:
    """The translation targets, one <st_xxx> each."""
    return [t[len("<st_") : -1] for t in tokens if t.startswith("<st_")]


def menus(tokens: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """The two dropdowns, read from the checkpoint rather than hardcoded.

    A model covering more languages or more translation targets needs no edit
    here: both menus are its own token list, labelled.

    Returns:
        (languages, targets), each a list of (menu label, ISO 639-3 code).
    """
    languages = _named(language_codes(tokens))
    targets = [
        (f"Translate to {LANGUAGE_NAMES.get(c, c)} ({c})", c)
        for _, c in _named(target_codes(tokens))
    ]
    return languages, targets


def read_audio(path: str) -> np.ndarray:
    """An audio file as the mono 16 kHz float array OWSM takes."""
    speech, _ = librosa.load(path, sr=SAMPLE_RATE)
    return speech


def pad(speech: np.ndarray) -> np.ndarray:
    """The one 30 s window OWSM decodes in a pass: the start of this audio.

    Shorter audio is zero-padded to the window the model is trained on, longer
    audio is cut to it. What librosa.util.fix_length does, in numpy, because
    reaching it through librosa.util imports scipy.ndimage for this one line.
    """
    window = SAMPLE_RATE * WINDOW_SECS
    return np.pad(speech[:window], (0, max(0, window - len(speech))))


def split_tokens(decoded: str, codes: Iterable[str]) -> Tuple[str, str]:
    """Separate OWSM's leading symbols from the text it decoded.

    The model writes the language and the task first, and can write a
    timestamp too. A symbol counts as the language only if the checkpoint
    lists it as one: "asr" is three lowercase letters as well, and a
    timestamp is a symbol like any other.
    """
    codes = frozenset(codes)
    language, rest = "", decoded.strip()
    while rest.startswith("<") and ">" in rest:
        symbol, rest = rest[1:].split(">", 1)
        if symbol in codes:
            language = symbol
        rest = rest.strip()
    return language, rest


# --- the local app, which is what `espnet demo` serves ---

# Worded like the CLI's own missing-soundfile message: what is absent, then
# the one command that fixes it.
GRADIO_MISSING = "gradio is not installed: pip install 'espnet[demo]'"
TITLE = "OWSM"
DESCRIPTION = """# OWSM

Transcribe, translate or identify the language of a recording with
[OWSM](https://www.wavlab.org/activities/2024/owsm/), the Open Whisper-style
Speech Models from [CMU WAVLab](https://www.wavlab.org/).

Served from your own machine by `espnet demo`. The language menu and the
translation targets below are this checkpoint's own.
"""


def load_gradio():
    """The gradio module, or None when it is not installed.

    Returning None rather than raising lets `espnet demo` report a missing
    package in the CLI's own one-line style, the way it reports a missing
    audio file, instead of ending in an ImportError traceback.
    """
    try:
        import gradio
    except ImportError:
        return None
    return gradio


def build_app(s2t, device: str = "cpu", model_tag: str = ""):
    """The Gradio app for an already loaded OWSM-CTC model.

    Args:
        s2t: a Speech2TextGreedySearch, loaded from the tag below.
        device: where it runs; long-form decoding batches on a GPU only.
        model_tag: shown in the page, so a demo of a different checkpoint
            says which one it is.

    Returns:
        A gradio Blocks, not yet launched.
    """
    gr = load_gradio()
    if gr is None:  # pragma: no cover - `espnet demo` checks this first
        raise ImportError(GRADIO_MISSING)

    languages, targets = menus(s2t.s2t_model.token_list)
    code_of_language = dict(languages)
    code_of_target = dict(targets)
    codes = frozenset(code_of_language.values())

    def detect(speech, task_sym):
        """The language OWSM-CTC names for the first window of this audio."""
        decoded = s2t(pad(speech), lang_sym="<nolang>", task_sym=task_sym)
        return split_tokens(decoded[0][0], codes)[0] or "eng"

    def predict(audio_path, language_label, task_label, long_form):
        if audio_path is None:
            raise gr.Error("Record or upload some audio first.")
        speech = read_audio(audio_path)
        if len(speech) > SAMPLE_RATE * MAX_SECS:
            gr.Warning(
                f"Only the first {MAX_SECS} s were decoded. "
                "`espnet asr` has no such limit."
            )
            speech = speech[: SAMPLE_RATE * MAX_SECS]

        chosen = None if language_label == DETECT else code_of_language[language_label]
        lang_sym = "<nolang>" if chosen is None else f"<{chosen}>"
        task_sym = (
            "<asr>" if task_label == ASR_LABEL else f"<st_{code_of_target[task_label]}>"
        )

        if long_form:
            # One 30 s pass first, only to name the language the rest is
            # decoded in; skipped when the user has already said what it is.
            detected = chosen or detect(speech, task_sym)
            text = s2t.decode_long_batched_buffered(
                speech,
                batch_size=1 if device == "cpu" else 8,
                context_len_in_secs=4,
                lang_sym=f"<{detected}>",
                task_sym=task_sym,
            )
        else:
            if len(speech) > SAMPLE_RATE * WINDOW_SECS:
                gr.Warning(
                    f"Only the first {WINDOW_SECS} s were decoded. "
                    "Tick Long-form for the whole recording."
                )
            decoded = s2t(pad(speech), lang_sym=lang_sym, task_sym=task_sym)
            detected, text = split_tokens(decoded[0][0], codes)
            detected = detected or chosen or ""
        return LANGUAGE_NAMES.get(detected, detected or "unknown"), text

    app = gr.Blocks(title=TITLE)
    with app:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["microphone", "upload"], type="filepath", label="Speech"
                )
                language = gr.Dropdown(
                    [DETECT] + [name for name, _ in languages],
                    value=DETECT,
                    label="Spoken language",
                )
                task = gr.Dropdown(
                    [ASR_LABEL] + [name for name, _ in targets],
                    value=ASR_LABEL,
                    label="Task",
                )
                long_form = gr.Checkbox(
                    label="Long-form",
                    info=f"Decode audio longer than {WINDOW_SECS} s in chunks",
                )
                button = gr.Button("Run", variant="primary")
            with gr.Column():
                detected = gr.Textbox(label="Language")
                text = gr.Textbox(label="Text", lines=8)
        button.click(predict, [audio, language, task, long_form], [detected, text])
        if model_tag:
            gr.Markdown(f"Model: `{model_tag}`, running on {device}.")
    return app
