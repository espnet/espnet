"""LJSpeech VITS: English text to speech.

Published as https://huggingface.co/spaces/espnet/ljspeech-vits; the source
lives in espnet, at egs2/ljspeech/tts1/demo. The recipe that trained the
checkpoint is the directory above this one.

One of the five demo Spaces this repository maintains; the others are
egs2/owsm_ctc_v4/s2t1/demo, egs2/owsm_v4/s2t1/demo,
egs2/universal_se_v1/enh1/demo and egs2/voxceleb/spk1/demo.
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from espnet2.bin.tts_inference import Text2Speech  # noqa: E402

# VITS synthesises far faster than real time, so the cap is not about the GPU
# slice: it is that a Space is a demo and not a text-to-audiobook service.
# Longer text is cut rather than refused, so pasting a whole page still
# returns something.
MAX_CHARS = 500
GPU_SECONDS = 60
MODEL_TAG = os.environ.get("TTS_MODEL_TAG", "espnet/kan-bayashi_ljspeech_vits")
# ZeroGPU attaches the GPU only while a @spaces.GPU function runs, so
# torch.cuda.is_available() is False here and asking it would pin the models to
# the CPU on the very hardware bought to run them. SPACES_ZERO_GPU is the
# runtime's own marker; `spaces` being importable is not, since anyone can
# install it.
ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
if os.environ.get("DEVICE"):
    DEVICE = os.environ["DEVICE"]
elif ZERO_GPU or torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


TITLE = "LJSpeech VITS"
DESCRIPTION = """# LJSpeech VITS

[VITS](https://arxiv.org/abs/2106.06103) is an end-to-end text-to-speech
model: text goes in and a waveform comes out, with no separate vocoder. This
checkpoint was trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
with [ESPnet](https://github.com/espnet/espnet), by the recipe at
[`egs2/ljspeech/tts1`](https://github.com/espnet/espnet/tree/master/egs2/ljspeech/tts1),
and speaks English in one voice.

The model is stochastic, so the same text twice gives two slightly different
readings.
"""
ARTICLE = """Model:
[`espnet/kan-bayashi_ljspeech_vits`](https://huggingface.co/espnet/kan-bayashi_ljspeech_vits).
Source of this Space:
[`egs2/ljspeech/tts1/demo`](https://github.com/espnet/espnet/tree/master/egs2/ljspeech/tts1/demo).

```bibtex
@article{hayashi2021espnet2,
  title={{ESP}net2-{TTS}: Extending the edge of {TTS} research},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Yoshimura, Takenori and
          Wu, Peter and Shi, Jiatong and Saeki, Takaaki and Ju, Yooncheol and
          Yasuda, Yusuke and Takamichi, Shinnosuke and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2110.07840},
  year={2021}
}
```"""


tts = Text2Speech.from_pretrained(MODEL_TAG, device=DEVICE)

if tts.use_spembs or tts.use_speech:
    # An x-vector model wants a speaker embedding and a GST model wants a
    # reference recording. Neither is something this interface collects, and
    # failing here names the reason instead of raising inside the first call.
    raise RuntimeError(
        f"{MODEL_TAG} needs a reference recording or a speaker embedding, "
        "which this demo does not take. Set TTS_MODEL_TAG to a checkpoint "
        "that synthesises from text alone."
    )

# The menus come from the checkpoint. LJSpeech is one English speaker, so both
# dropdowns are empty here and stay hidden; a multi-speaker or multi-lingual
# checkpoint reached through TTS_MODEL_TAG fills them without an edit.
SPEAKERS = list(range(tts.tts.spks)) if tts.use_sids else []
LANGUAGES = list(range(tts.tts.langs)) if tts.use_lids else []
# Duration scaling is offered by VITS, FastSpeech and FastSpeech2 and not by
# Tacotron 2 or Transformer-TTS; the decoding config the model built for
# itself is what says which of them this is.
HAS_SPEED = "alpha" in tts.decode_conf

EXAMPLE_TEXT = (
    "ESPnet is an end-to-end speech processing toolkit, "
    "and this sentence was spoken by a model trained with it."
)


@spaces.GPU(duration=GPU_SECONDS)
def predict(text, speaker, language, speed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type something for the model to say first.")
    if len(text) > MAX_CHARS:
        gr.Warning(
            f"Only the first {MAX_CHARS} characters were read; "
            f"the text is {len(text)}. Run the app yourself for more - "
            "the model has no such limit."
        )
        text = text[:MAX_CHARS]

    conditioning = {}
    if SPEAKERS:
        conditioning["sids"] = np.array([int(speaker)])
    if LANGUAGES:
        conditioning["lids"] = np.array([int(language)])
    # alpha scales the predicted durations, so it is the reciprocal of speed:
    # alpha 2 is twice as long and half as fast.
    decode_conf = {"alpha": 1.0 / speed} if HAS_SPEED else None

    wav = tts(text, decode_conf=decode_conf, **conditioning)["wav"]
    return tts.fs, wav.view(-1).cpu().numpy()


EXAMPLES = [
    [EXAMPLE_TEXT, SPEAKERS[0] if SPEAKERS else None, None, 1.0],
]


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Text",
                lines=4,
                info=f"Up to {MAX_CHARS} characters",
            )
            speaker = gr.Dropdown(
                SPEAKERS,
                value=SPEAKERS[0] if SPEAKERS else None,
                label="Speaker",
                visible=bool(SPEAKERS),
            )
            language = gr.Dropdown(
                LANGUAGES,
                value=LANGUAGES[0] if LANGUAGES else None,
                label="Language",
                visible=bool(LANGUAGES),
            )
            speed = gr.Slider(
                0.5,
                2.0,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1 is the pace the model was trained at",
                visible=HAS_SPEED,
            )
            button = gr.Button("Speak", variant="primary")
        with gr.Column():
            audio = gr.Audio(label="Speech", type="numpy")
    button.click(predict, [text, speaker, language, speed], audio)
    gr.Examples(
        EXAMPLES,
        inputs=[text, speaker, language, speed],
        outputs=audio,
        fn=predict,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
