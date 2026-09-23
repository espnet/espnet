#!/usr/bin/env python3
"""The browser demo of a speech-to-text model: what `espnet demo` serves.

The page is the checkpoint's rather than OWSM's: the language menu, the
translation targets, the window it decodes in a pass and its own spelling of
"no language given" all come off the model that was loaded, and a checkpoint
with `<pr>` in its token list - POWSM, the phonetic model built on OWSM - is
offered phone recognition beside transcription.

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

# Two limits, from two different places, which is why they are named apart.
SAMPLE_RATE = 16000
# The model's: what OWSM was trained on. Longer audio is decoded in chunks,
# and a checkpoint trained on a different length would change this number.
WINDOW_SECS = 30
# The demo's: a browser demo answers while someone waits, and two minutes of
# audio is already a wait. It has nothing to do with OWSM - neither the model
# nor `espnet transcribe` has such a limit - and it is the same number for any
# model this command grew to serve. Both Spaces refuse audio longer than
# this; `espnet demo` decodes the first two minutes and says so.
MAX_SECS = 120
DETECT = "Detect automatically"
ASR_LABEL = "Transcribe"
# POWSM, the phonetic model built on OWSM, answers <pr> with the phones it
# hears. A checkpoint that has the symbol gets the option; OWSM does not have
# it and its menu is unchanged.
PHONES_LABEL = "Recognise phones"
PHONE_TASK = "<pr>"
# POWSM's other two tasks take the audio and something written with it: the
# words that were said, to be answered with phones, or the phones, to be
# answered with words. A checkpoint that has the symbol gets the entry, and
# the box to type the input into.
G2P_LABEL = "Phones for text you give (G2P)"
G2P_TASK = "<g2p>"
P2G_LABEL = "Text for phones you give (P2G)"
P2G_TASK = "<p2g>"
PROMPT_TASKS = {G2P_LABEL: G2P_TASK, P2G_LABEL: P2G_TASK}
PROMPT_LABELS = {
    G2P_LABEL: "The words that were said",
    P2G_LABEL: "The phones, spaced or between slashes",
}
# Shown when the loaded checkpoint offers phones, because the same page then
# also offers Transcribe on a model built for something else. POWSM's author
# asked for this to be said where someone would read it: its English ASR is
# weak - a text normalisation problem the authors have since retrained - and
# a page that offers the button without the caveat invites the wrong reading.
PHONE_MODEL_NOTE = (
    "This checkpoint is a phonetic model. **Recognise phones** is what it is "
    "for; its transcription is weaker than a model trained for text, and on "
    "[POWSM](https://huggingface.co/espnet/powsm) in particular the English "
    "ASR suffers from a text normalisation problem — the authors have since "
    "published a retrained variant in that repository's `textnorm_retrained` "
    "folder."
)
PHONE = re.compile(r"/([^/]+)/")

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


def read_audio(path: str, rate: int = SAMPLE_RATE) -> np.ndarray:
    """An audio file as the mono float array a model takes, at its own rate."""
    speech, _ = librosa.load(path, sr=rate)
    return speech


def pad(
    speech: np.ndarray,
    window_secs: int = WINDOW_SECS,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """The one window the model decodes in a pass: the start of this audio.

    Shorter audio is zero-padded to the window the model is trained on, longer
    audio is cut to it. What librosa.util.fix_length does, in numpy, because
    reaching it through librosa.util imports scipy.ndimage for this one line.

    The default is OWSM's 30 s, which is what the two Space apps pass and what
    this module meant for as long as it served one model. POWSM's window is
    20 s, and padding it to 30 would be 10 s of silence for the model to
    hallucinate over.
    """
    window = rate * window_secs
    return np.pad(speech[:window], (0, max(0, window - len(speech))))


def window_secs(s2t) -> int:
    """The window this checkpoint was trained on, in seconds."""
    conf = getattr(s2t, "preprocessor_conf", None) or {}
    return int(conf.get("speech_length", WINDOW_SECS))


def sample_rate(s2t) -> int:
    """The rate this checkpoint's audio is read at.

    Like the window: the module's constant is what OWSM uses and what the two
    Space apps pass, and a checkpoint that says otherwise is believed. Nothing
    in this page should hold a number a model could tell it.
    """
    rate = getattr(s2t, "sample_rate", None)
    if rate:
        return int(rate)
    conf = getattr(s2t, "preprocessor_conf", None) or {}
    return int(conf.get("fs", SAMPLE_RATE))


def phone_task(tokens: Sequence[str]) -> bool:
    """Whether this checkpoint answers the phone recognition symbol."""
    return PHONE_TASK in tokens


def prompt_tasks(tokens: Sequence[str]) -> List[str]:
    """The labels of the tasks this checkpoint takes written input for.

    POWSM has `<g2p>` and `<p2g>`; OWSM has neither, and its page is
    unchanged.
    """
    return [label for label, task in PROMPT_TASKS.items() if task in tokens]


def as_phones(text: str) -> str:
    """Phones the way POWSM was trained to read them, from either spelling.

    Its training data writes each phone between slashes - /p//h//o/ - so that
    a phone spelled like a BPE token is still one token. The page shows them
    spaced, which is what anything counting or aligning them wants, and a
    person typing them in will type them that way too.
    """
    text = text.strip()
    if not text:
        return text
    if "/" in text:
        return text
    return "/" + "//".join(text.split()) + "/"


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

    Only gradio's own absence answers None. An installed gradio that fails on
    one of its dependencies is a broken install, not a missing extra, and
    saying "gradio is not installed" would hide the name of what is.
    """
    try:
        import gradio
    except ModuleNotFoundError as e:
        if e.name != "gradio":
            raise
        return None
    return gradio


def build_app(s2t, device: str = "cpu", model_tag: str = "", wrap=None):
    """The Gradio app for an already loaded OWSM-CTC model.

    Args:
        s2t: a Speech2Text, loaded from the tag below. It is decoded with
            best_path() and decode_long(), not by calling it: a browser
            demo answers while someone waits, and a search on the CTC head
            is an order of magnitude slower for no gain here.
        device: where it runs; long-form decoding batches on a GPU only.
        model_tag: shown in the page, so a demo of a different checkpoint
            says which one it is.
        wrap: applied to the function behind the Run button, for a caller
            that has to say something about how it runs. A Hugging Face
            Space on ZeroGPU has no GPU except inside a function the
            `spaces` package has decorated, and that decorator belongs to
            the Space rather than to this module - so the Space passes it
            in and gets the same page as everyone else.

    Returns:
        A gradio Blocks, not yet launched.
    """
    gr = load_gradio()
    if gr is None:  # pragma: no cover - `espnet demo` checks this first
        raise ImportError(GRADIO_MISSING)

    tokens = s2t.s2t_model.token_list
    languages, targets = menus(tokens)
    code_of_language = dict(languages)
    code_of_target = dict(targets)
    codes = frozenset(code_of_language.values())
    # the checkpoint's own window and its own spelling of "no language given":
    # OWSM is 30 s and <nolang>, POWSM is 20 s and <unk>
    window = window_secs(s2t)
    rate = sample_rate(s2t)
    phones = phone_task(tokens)
    prompted = prompt_tasks(tokens)
    try:
        nolang = s2t.no_language()
    except ValueError:
        # A checkpoint with no symbol for "work it out yourself" cannot be
        # asked to detect: the menu then has no Detect entry and opens on a
        # language instead. Better than a page that raises on every Run.
        nolang = None

    def detect(speech, task_sym):
        """The language the model names for the first window of this audio."""
        decoded = s2t.decode_window(pad(speech, window, rate), nolang, task_sym)
        return split_tokens(decoded, codes)[0] or "eng"

    def chosen_language(label):
        """The code the menu is showing, or None when it says Detect."""
        return None if label == DETECT else code_of_language[label]

    def predict(audio_path, language_label, task_label, long_form, prompt=""):
        if audio_path is None:
            raise gr.Error("Record or upload some audio first.")
        speech = read_audio(audio_path, rate)
        if len(speech) > rate * MAX_SECS:
            gr.Warning(
                f"Only the first {MAX_SECS} s were decoded. "
                "`espnet transcribe` has no such limit."
            )
            speech = speech[: rate * MAX_SECS]

        chosen = chosen_language(language_label)
        lang_sym = nolang if chosen is None else f"<{chosen}>"
        text_prev = "<na>"
        if task_label == ASR_LABEL:
            task_sym = "<asr>"
        elif task_label == PHONES_LABEL:
            task_sym = PHONE_TASK
        elif task_label in PROMPT_TASKS:
            task_sym = PROMPT_TASKS[task_label]
            if not (prompt or "").strip():
                raise gr.Error(f"{PROMPT_LABELS[task_label]}: this task needs it.")
            text_prev = as_phones(prompt) if task_sym == P2G_TASK else prompt.strip()
        else:
            task_sym = f"<st_{code_of_target[task_label]}>"

        if task_sym in PROMPT_TASKS.values() and long_form:
            # The written input is one utterance's, and the windows after the
            # first would each be given the whole of it again.
            gr.Warning(f"{task_label} reads one window; Long-form was ignored.")
            long_form = False

        if long_form:
            # One window first, only to name the language the rest is decoded
            # in; skipped when the user has already said what it is.
            detected = chosen or detect(speech, task_sym)
            phone_pass = task_sym == PHONE_TASK and not s2t.ctc_only
            if not phone_pass:
                text = " ".join(
                    segment
                    for _, _, segment in s2t.decode_long(
                        speech,
                        batch_size=1 if device == "cpu" else 8,
                        context_len_in_secs=4,
                        lang_sym=f"<{detected}>",
                        task_sym=task_sym,
                    )
                )
            else:
                # The exception is a task rather than a kind of model. An
                # encoder-decoder checkpoint segments a long recording by the
                # timestamps it writes, which is what OWSM does well; POWSM
                # asked for phones instead fills the padding at the end of a
                # window with repetitions of what it already said. That one
                # combination goes window by window.
                step = rate * window
                text = " ".join(
                    split_tokens(
                        s2t.decode_window(
                            speech[at : at + step], f"<{detected}>", task_sym
                        ),
                        codes,
                    )[1]
                    for at in range(0, len(speech), step)
                )
        else:
            if len(speech) > rate * window:
                gr.Warning(
                    f"Only the first {window} s were decoded. "
                    "Tick Long-form for the whole recording."
                )
            decoded = s2t.decode_window(
                pad(speech, window, rate), lang_sym, task_sym, text_prev
            )
            detected, text = split_tokens(decoded, codes)
            detected = detected or chosen or ""
        if task_sym in (PHONE_TASK, G2P_TASK):
            # POWSM writes each phone between slashes, so that a phone spelled
            # like a BPE token is still one token. The page shows them spaced,
            # which is the form anything counting or aligning them wants.
            text = " ".join(PHONE.findall(text)) or text
        return LANGUAGE_NAMES.get(detected, detected or "unknown"), text

    if wrap is not None:
        predict = wrap(predict)

    app = gr.Blocks(title=TITLE)
    with app:
        gr.Markdown(DESCRIPTION)
        if phones:
            gr.Markdown(PHONE_MODEL_NOTE)
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["microphone", "upload"], type="filepath", label="Speech"
                )
                choices = ([DETECT] if nolang else []) + [n for n, _ in languages]
                language = gr.Dropdown(
                    choices,
                    value=choices[0],
                    label="Spoken language",
                )
                task = gr.Dropdown(
                    [ASR_LABEL]
                    + ([PHONES_LABEL] if phones else [])
                    + prompted
                    + [name for name, _ in targets],
                    value=ASR_LABEL,
                    label="Task",
                )
                # shown only for the tasks that read it, since for the rest
                # there is nothing to type and a box invites typing anyway
                prompt = gr.Textbox(
                    label="Written input",
                    lines=2,
                    visible=False,
                )
                long_form = gr.Checkbox(
                    label="Long-form",
                    info=f"Decode audio longer than {window} s in chunks",
                )
                button = gr.Button("Run", variant="primary")
            with gr.Column():
                detected = gr.Textbox(label="Language")
                text = gr.Textbox(label="Text", lines=8)
        if prompted:

            def show_prompt(task_label):
                """The box, labelled for the task that reads it."""
                return gr.update(
                    visible=task_label in PROMPT_TASKS,
                    label=PROMPT_LABELS.get(task_label, "Written input"),
                )

            task.change(show_prompt, task, prompt)
        button.click(
            predict, [audio, language, task, long_form, prompt], [detected, text]
        )
        if model_tag:
            gr.Markdown(f"Model: `{model_tag}`, running on {device}.")
    return app
