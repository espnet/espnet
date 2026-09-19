"""OWSM v4: speech recognition, translation, language ID and prompting.

Published as https://huggingface.co/spaces/espnet/owsm-v4; the source lives in
espnet, at egs2/owsm_v4/s2t1/demo. Its encoder-only sibling is
egs2/owsm_ctc_v4/s2t1/demo.

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

# Everything this app shares with egs2/owsm_ctc_v4/s2t1/demo and with
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
)
from espnet2.bin.s2t_inference import Speech2Text  # noqa: E402
from espnet2.bin.s2t_inference_language import Speech2Language  # noqa: E402

# ZeroGPU gives a decorated call a fixed slice of GPU time and kills it at the
# end, so the demo asks for a slice and refuses audio it could not finish in
# one (MAX_SECS). The Space this replaces limited the input the same way.
GPU_SECONDS = 120
MODEL_TAG = os.environ.get("OWSM_MODEL_TAG", "espnet/owsm_v4_medium_1B")
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

# The menus come from the checkpoint: OWSM's token list holds every language
# between <nolang> and <asr>, and one <st_xxx> per translation target.
LANGUAGES, TARGETS = menus(s2t.s2t_model.token_list)
CODE_OF_LANGUAGE = dict(LANGUAGES)
CODE_OF_TARGET = dict(TARGETS)


@spaces.GPU(duration=GPU_SECONDS)
def predict(audio_path, language_label, task_label, long_form, prompt):
    speech = _read(audio_path)
    if language_label == DETECT:
        code = s2l(pad(speech[: SAMPLE_RATE * WINDOW_SECS]))[0][0].strip()[1:-1]
    else:
        code = CODE_OF_LANGUAGE[language_label]
    lang_sym = f"<{code}>"
    task_sym = (
        "<asr>" if task_label == ASR_LABEL else f"<st_{CODE_OF_TARGET[task_label]}>"
    )
    prompt = prompt.strip()

    if long_form:
        utterances = s2t.decode_long(
            speech,
            init_text=prompt or None,
            lang_sym=lang_sym,
            task_sym=task_sym,
        )
        text = "\n".join(
            f"[{float(start):6.2f} - {float(end):6.2f}] {line}"
            for start, end, line in utterances
        )
    else:
        if len(speech) > SAMPLE_RATE * WINDOW_SECS:
            gr.Warning(
                f"Only the first {WINDOW_SECS} s were decoded. "
                "Tick Long-form for the whole recording."
            )
        text = s2t(
            pad(speech[: SAMPLE_RATE * WINDOW_SECS]),
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
