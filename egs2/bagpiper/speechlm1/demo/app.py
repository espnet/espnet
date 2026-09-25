"""Bagpiper: what a recording is, and a recording of what you describe.

Published as https://huggingface.co/spaces/espnet/bagpiper; the source lives
in espnet, at egs2/bagpiper/speechlm1/demo.

The other demos go one way. This model goes both, through the same thing in
the middle: it maps audio to a rich caption - a full description of what is
there - and a description back to audio. So the page is a round trip. Ask
what a recording is, and the answer lands in the box that renders it back;
edit a word of the description and hear what changes.

Models: espnet/bagpiper-sft does both directions, espnet/bagpiper-tts-sft is
the speech-focused one (BAGPIPER_MODEL_TAG switches). Either is an 8B model
with an audio encoder and a codec attached; BAGPIPER_URL points the page at
one that is already served instead of loading it here.
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
import tempfile  # noqa: E402

import gradio as gr  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402

from espnet2.bin.speechlm_inference import (  # noqa: E402
    Decoding,
    from_pretrained,
    from_server,
    split_thinking,
)

# The model was trained on clips of up to 30 seconds, so a longer one is
# outside what it has seen rather than merely slower. Rendering is the long
# half of the slice: up to 2048 decode steps of audio.
MAX_SECS = 30
GPU_SECONDS = 180
MODEL_TAG = os.environ.get("BAGPIPER_MODEL_TAG", "espnet/bagpiper-sft")
# An address, if this Space is a front for a model served elsewhere; empty
# means load the checkpoint here.
SERVER = os.environ.get("BAGPIPER_URL", "")
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

EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)
EXAMPLE_QUESTION = "What is in this recording? Describe the voice and the room."
EXAMPLE_SCENE = (
    "A clear, friendly female voice, close-miked in a quiet room, says: "
    "'Hello, how are you today?'. She speaks at a relaxed, natural pace with "
    "a warm tone and no background noise."
)

TITLE = "Bagpiper"
DESCRIPTION = """# Bagpiper

An 8B audio model that works in both directions through the same
representation: a **rich caption**, its own full description of a sound.
Ask it what a recording is, and it writes one. Give it one, and it renders
the audio.

That is why this page is a loop rather than two demos. What it heard lands
in the box below, and you can render it back - or change a word first and
hear what that does.

It covers speech, music and environmental sound, and mixtures of them.
[Paper](https://arxiv.org/abs/2602.05220) ·
[Bagpiper-TTS](https://arxiv.org/abs/2606.22811)
"""
ARTICLE = """**Describing is not instructing.** The generator renders a scene
you describe, with any speech quoted inside the description - "A calm male
voice says: 'your package arrives Tuesday'. No background noise." - because
that is the shape it was trained on. "Read this aloud: ..." returns audio,
but not a faithful reading. For a sentence read as written, the
[ljspeech-vits](https://huggingface.co/spaces/espnet/ljspeech-vits) Space is
text-to-speech proper.

The model thinks before it answers in both directions, so the reasoning is
shown apart from the answer where it is marked.

`espnet describe audio.wav` and `espnet render "..."` are the same two things
from a terminal, and they pipe into each other. Source of this Space:
[`egs2/bagpiper/speechlm1/demo`](https://github.com/espnet/espnet/tree/master/egs2/bagpiper/speechlm1/demo).
"""  # noqa: E501 - markdown links, and breaking a URL breaks the link

model = from_server(SERVER) if SERVER else from_pretrained(MODEL_TAG, device=DEVICE)


def _seconds(path):
    if path is None:
        raise gr.Error("Record or upload some audio first.")
    seconds = librosa.get_duration(path=path)
    if seconds > MAX_SECS:
        raise gr.Error(
            f"This demo takes up to {MAX_SECS} s; that file is {seconds:.0f} s. "
            "The model was trained on clips that long, so a longer one is "
            "outside what it has seen - cut it, or run the app yourself."
        )
    return seconds


@spaces.GPU(duration=GPU_SECONDS)
def describe(audio_path, prompt):
    _seconds(audio_path)
    answer = model.describe(audio_path, prompt=prompt or "What is in this audio?")
    thinking, said = split_thinking(answer)
    # the answer is what renders back, so it is what fills the scene box
    return thinking, said, said


@spaces.GPU(duration=GPU_SECONDS)
def render(scene, cfg):
    if not (scene or "").strip():
        raise gr.Error("Describe the audio you want first.")
    audio, text = model.render(scene, decoding=Decoding(cfg=cfg))
    thinking, said = split_thinking(text)
    if audio is None:
        gr.Warning(
            "The model answered with text and no audio, which it is allowed "
            "to do. Describing a scene rather than asking for one usually "
            "fixes it."
        )
        return None, thinking, said
    # the model hands back a WAV; gradio plays a file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
    return f.name, thinking, said


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. What is this?")
            audio_in = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Audio"
            )
            question = gr.Textbox(
                label="Ask about it",
                value=EXAMPLE_QUESTION,
                lines=2,
            )
            ask = gr.Button("Describe", variant="primary")
            answer = gr.Textbox(label="Answer", lines=6)
            with gr.Accordion("How it got there", open=False):
                thinking_in = gr.Markdown()
        with gr.Column():
            gr.Markdown("### 2. Render it back")
            scene = gr.Textbox(
                label="The audio to render, described",
                value=EXAMPLE_SCENE,
                lines=6,
            )
            cfg = gr.Slider(
                1.0,
                5.0,
                value=3.0,
                step=0.5,
                label="Guidance",
                info="how closely the audio follows the description",
            )
            make = gr.Button("Render", variant="primary")
            audio_out = gr.Audio(label="Rendered", type="filepath")
            plan = gr.Textbox(label="What it decided to render", lines=4)
            with gr.Accordion("How it got there", open=False):
                thinking_out = gr.Markdown()

    ask.click(describe, [audio_in, question], [thinking_in, answer, scene])
    make.click(render, [scene, cfg], [audio_out, thinking_out, plan])
    gr.Examples(
        [[EXAMPLE_WAV, EXAMPLE_QUESTION]],
        inputs=[audio_in, question],
        outputs=[thinking_in, answer, scene],
        fn=describe,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
