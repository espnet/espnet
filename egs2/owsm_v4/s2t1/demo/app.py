"""OWSM v4: speech recognition, translation, language ID and prompting.

Published as https://huggingface.co/spaces/espnet/owsm-v4; the source lives in
espnet, at egs2/owsm_v4/s2t1/demo. Its encoder-only sibling is
egs2/owsm_ctc_v4/s2t1/demo.

Both demos offer the same tasks - speech recognition, any-to-any speech
translation, language identification and long-form decoding - so that the two
models can be compared on the same audio.
"""

import os
import re

import gradio as gr
import librosa
import torch

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.bin.s2t_inference_language import Speech2Language

try:  # ZeroGPU's decorator exists only on Hugging Face's runners
    import spaces

    ZERO_GPU = True
except ImportError:  # running locally: the decorator does nothing
    ZERO_GPU = False

    class spaces:  # noqa: N801 - stands in for the module
        @staticmethod
        def GPU(func):
            return func


SAMPLE_RATE = 16000
WINDOW_SECS = 30  # what OWSM is trained on; longer audio is decoded in chunks
MODEL_TAG = os.environ.get("OWSM_MODEL_TAG", "espnet/owsm_v4_medium_1B")
# On ZeroGPU the GPU is attached only while a @spaces.GPU function runs, so
# torch.cuda.is_available() is False here and asking it would pin the models to
# the CPU on the very hardware bought to run them.
if os.environ.get("DEVICE"):
    DEVICE = os.environ["DEVICE"]
elif ZERO_GPU or os.environ.get("SPACES_ZERO_GPU"):
    DEVICE = "cuda"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
DETECT = "Detect automatically"


EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)


def _names(codes):
    """Menu labels for the model's own language codes, sorted by label."""
    return sorted((f"{LANGUAGE_NAMES.get(c, c)} ({c})", c) for c in codes)


def _language_codes(tokens):
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


def _target_codes(tokens):
    """The translation targets, one <st_xxx> each."""
    return [t[len("<st_") : -1] for t in tokens if t.startswith("<st_")]


def _read(path):
    if path is None:
        raise gr.Error("Record or upload some audio first.")
    speech, _ = librosa.load(path, sr=SAMPLE_RATE)
    return speech


def _pad(speech):
    """OWSM is trained on a fixed 30 s window."""
    return librosa.util.fix_length(speech, size=SAMPLE_RATE * WINDOW_SECS)


TITLE = "OWSM v4"
DESCRIPTION = """# OWSM v4

[OWSM](https://www.wavlab.org/activities/2024/owsm/) is a series of Open
Whisper-style Speech Models from [CMU WAVLab](https://www.wavlab.org/),
reproducing Whisper-style training on public data with
[ESPnet](https://github.com/espnet/espnet). This is the encoder-decoder v4
medium model: it transcribes, translates, identifies the language, decodes
long-form audio, and can be steered with a text prompt.

The encoder-only CTC sibling, which is faster, is at
[espnet/owsm-ctc-v4](https://huggingface.co/spaces/espnet/owsm-ctc-v4).
"""
ARTICLE = """Model:
[`espnet/owsm_v4_medium_1B`](https://huggingface.co/espnet/owsm_v4_medium_1B)
(CC-BY-4.0). Source of this Space:
[`egs2/owsm_v4/s2t1/demo`](https://github.com/espnet/espnet/tree/master/egs2/owsm_v4/s2t1/demo).

OWSM has not been evaluated on every task it supports; with limited training
data it may do poorly in some languages.

```bibtex
@inproceedings{owsm-v4,
  title={{OWSM} v4: Improving Open Whisper-Style Speech Models via Data Scaling
         and Cleaning},
  author={Yifan Peng and Shakeel Muhammad and Yui Sudo and William Chen and
          Jinchuan Tian and Chyi-Jiunn Lin and Shinji Watanabe},
  booktitle={Proc. Interspeech},
  year={2025}
}
```"""


s2t = Speech2Text.from_pretrained(
    MODEL_TAG,
    device=DEVICE,
    beam_size=5,
    ctc_weight=0.0,
    maxlenratio=0.0,
    lang_sym="<eng>",
    task_sym="<asr>",
    predict_time=False,
)
s2l = Speech2Language.from_pretrained(MODEL_TAG, device=DEVICE, nbest=1)

LANGUAGES = _names(_language_codes(s2t.s2t_model.token_list))
TARGETS = [
    (f"Translate to {LANGUAGE_NAMES.get(c, c)} ({c})", c)
    for _, c in _names(_target_codes(s2t.s2t_model.token_list))
]
ASR_LABEL = "Transcribe"


@spaces.GPU
def predict(audio_path, language_label, task_label, long_form, prompt):
    speech = _read(audio_path)
    if language_label == DETECT:
        code = s2l(_pad(speech[: SAMPLE_RATE * WINDOW_SECS]))[0][0].strip()[1:-1]
    else:
        code = dict(LANGUAGES)[language_label]
    lang_sym = f"<{code}>"
    task_sym = (
        "<asr>" if task_label == ASR_LABEL else f"<st_{dict(TARGETS)[task_label]}>"
    )
    prompt = prompt.strip()

    if long_form:
        utterances = s2t.decode_long(
            speech,
            init_text=prompt or None,
            lang_sym=lang_sym,
            task_sym=task_sym,
        )
        text = "\n".join(f"[{start} - {end}] {line}" for start, end, line in utterances)
    else:
        text = s2t(
            _pad(speech[: SAMPLE_RATE * WINDOW_SECS]),
            prompt or "<na>",
            lang_sym=lang_sym,
            task_sym=task_sym,
        )[0][-2]
    return LANGUAGE_NAMES.get(code, code), text


EXAMPLES = [[EXAMPLE_WAV, DETECT, ASR_LABEL, False, ""]]


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Speech"
            )
            language = gr.Dropdown(
                [DETECT] + [name for name, _ in LANGUAGES],
                value=DETECT,
                label="Spoken language",
            )
            task = gr.Dropdown(
                [ASR_LABEL] + [name for name, _ in TARGETS],
                value=ASR_LABEL,
                label="Task",
            )
            long_form = gr.Checkbox(
                label="Long-form",
                info=f"Decode audio longer than {WINDOW_SECS} s in chunks",
            )
            prompt = gr.Textbox(
                label="Text prompt (optional)",
                info="Decoding is conditioned on this text",
            )
            button = gr.Button("Run", variant="primary")
        with gr.Column():
            detected = gr.Textbox(label="Language")
            text = gr.Textbox(label="Text", lines=8)
    button.click(predict, [audio, language, task, long_form, prompt], [detected, text])
    gr.Examples(
        EXAMPLES,
        inputs=[audio, language, task, long_form, prompt],
        outputs=[detected, text],
        fn=predict,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
