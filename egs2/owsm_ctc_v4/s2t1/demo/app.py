"""OWSM-CTC v4: speech recognition, translation and language ID.

Published as https://huggingface.co/spaces/espnet/owsm-ctc-v4; the source lives in
espnet, at egs2/owsm_ctc_v4/s2t1/demo. Its autoregressive sibling is
egs2/owsm_v4/s2t1/demo.

Both demos offer the same tasks - speech recognition, any-to-any speech
translation, language identification and long-form decoding - so that the two
models can be compared on the same audio.
"""

# The ZeroGPU package patches torch as it is imported, so it has to come
# first - before torch, and before anything that imports torch.
try:  # only Hugging Face's runners have it
    import spaces  # isort: skip
except ImportError:  # running elsewhere: the decorator does nothing

    class spaces:  # noqa: N801 - stands in for the module
        @staticmethod
        def GPU(func=None, **kwargs):
            return func if func is not None else (lambda f: f)


import os  # noqa: E402

import gradio as gr  # noqa: E402

# Everything this app shares with egs2/owsm_v4/s2t1/demo and with
# `espnet demo`: the language table, the menus a checkpoint describes, and the
# reading and padding of audio. Shipped in the espnet release pinned by
# requirements.txt, so the two Spaces and the command cannot drift apart.
from espnet2.bin.demo import (  # noqa: E402
    ASR_LABEL,
    DETECT,
    LANGUAGE_NAMES,
    MAX_SECS,
    SAMPLE_RATE,
    WINDOW_SECS,
    default_device,
    menus,
    pad,
    read_audio,
    split_tokens,
)
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch  # noqa: E402

# ZeroGPU gives a decorated call a fixed slice of GPU time and kills it at the
# end, so the demo asks for a slice and refuses audio it could not finish in
# one (MAX_SECS). The Space this replaces limited the input the same way.
GPU_SECONDS = 120
MODEL_TAG = os.environ.get("OWSM_MODEL_TAG", "espnet/owsm_ctc_v4_1B")
# ZeroGPU attaches the GPU only while a @spaces.GPU function runs, so
# torch.cuda.is_available() is False here and asking it would pin the models to
# the CPU on the very hardware bought to run them. SPACES_ZERO_GPU is the
# runtime's own marker; `spaces` being importable is not, since anyone can
# install it.
ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
if os.environ.get("DEVICE"):
    DEVICE = os.environ["DEVICE"]
elif ZERO_GPU:
    DEVICE = "cuda"
else:
    DEVICE = default_device()


EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)


def _read(path):
    if path is None:
        raise gr.Error("Record or upload some audio first.")
    speech = read_audio(path)
    seconds = len(speech) / SAMPLE_RATE
    if seconds > MAX_SECS:
        raise gr.Error(
            f"This demo takes up to {MAX_SECS} s; that file is {seconds:.0f} s. "
            "Run the app yourself for longer audio - the model has no such limit."
        )
    return speech


TITLE = "OWSM-CTC v4"
DESCRIPTION = """# OWSM-CTC v4

[OWSM-CTC](https://aclanthology.org/2024.acl-long.549/) is an encoder-only
speech foundation model from [CMU WAVLab](https://www.wavlab.org/), trained on
320k hours of public audio with [ESPnet](https://github.com/espnet/espnet).
One encoder pass per 30 s window, no beam search: it transcribes, translates
and identifies the language, and it is fast.

Its autoregressive sibling, which adds text prompting, runs the same tasks:
[espnet/owsm-v4](https://huggingface.co/spaces/espnet/owsm-v4).
"""
ARTICLE = """Model:
[`espnet/owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B)
(CC-BY-4.0). Source of this Space:
[`egs2/owsm_ctc_v4/s2t1/demo`](https://github.com/espnet/espnet/tree/master/egs2/owsm_ctc_v4/s2t1/demo).

```bibtex
@inproceedings{owsm-ctc,
  title={{OWSM-CTC}: An Open Encoder-Only Speech Foundation Model for Speech
         Recognition, Translation, and Language Identification},
  author={Yifan Peng and Yui Sudo and Muhammad Shakeel and Shinji Watanabe},
  booktitle={Proc. ACL},
  year={2024}
}
```"""


s2t = Speech2TextGreedySearch.from_pretrained(
    MODEL_TAG,
    device=DEVICE,
    generate_interctc_outputs=False,
    lang_sym="<nolang>",
    task_sym="<asr>",
)

# The menus come from the checkpoint: OWSM's token list holds every language
# between <nolang> and <asr>, and one <st_xxx> per translation target.
LANGUAGES, TARGETS = menus(s2t.s2t_model.token_list)
LANGUAGE_CODES = frozenset(code for _, code in LANGUAGES)
CODE_OF_LANGUAGE = dict(LANGUAGES)
CODE_OF_TARGET = dict(TARGETS)


def _detect(speech, task_sym):
    """The language OWSM-CTC names for the first window of this audio."""
    decoded = s2t(
        pad(speech[: SAMPLE_RATE * WINDOW_SECS]),
        lang_sym="<nolang>",
        task_sym=task_sym,
    )
    return split_tokens(decoded[0][0], LANGUAGE_CODES)[0] or "eng"


@spaces.GPU(duration=GPU_SECONDS)
def predict(audio_path, language_label, task_label, long_form):
    speech = _read(audio_path)
    lang_sym = (
        "<nolang>"
        if language_label == DETECT
        else f"<{CODE_OF_LANGUAGE[language_label]}>"
    )
    task_sym = (
        "<asr>" if task_label == ASR_LABEL else f"<st_{CODE_OF_TARGET[task_label]}>"
    )

    chosen = None if language_label == DETECT else CODE_OF_LANGUAGE[language_label]
    if long_form:
        # One 30 s pass first, only to name the language the rest is decoded
        # in; skipped when the user has already said what it is.
        detected = chosen or _detect(speech, task_sym)
        text = s2t.decode_long_batched_buffered(
            speech,
            batch_size=1 if DEVICE == "cpu" else 8,
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
        decoded = s2t(
            pad(speech[: SAMPLE_RATE * WINDOW_SECS]),
            lang_sym=lang_sym,
            task_sym=task_sym,
        )
        detected, text = split_tokens(decoded[0][0], LANGUAGE_CODES)
        detected = detected or chosen or ""
    return LANGUAGE_NAMES.get(detected, detected or "unknown"), text


EXAMPLES = [[EXAMPLE_WAV, DETECT, ASR_LABEL, False]]


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
            button = gr.Button("Run", variant="primary")
        with gr.Column():
            detected = gr.Textbox(label="Language")
            text = gr.Textbox(label="Text", lines=8)
    button.click(predict, [audio, language, task, long_form], [detected, text])
    gr.Examples(
        EXAMPLES,
        inputs=[audio, language, task, long_form],
        outputs=[detected, text],
        fn=predict,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
