"""Universal speech enhancement with USES.

Published as https://huggingface.co/spaces/espnet/universal-se; the source
lives in espnet, at egs2/universal_se_v1/enh1/demo. The recipe that trained
the checkpoint is the directory above this one.

One of the five demo Spaces this repository maintains; the others are
egs2/owsm_ctc_v4/s2t1/demo, egs2/owsm_v4/s2t1/demo, egs2/ljspeech/tts1/demo
and egs2/voxceleb/spk1/demo.
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
import re  # noqa: E402

import gradio as gr  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402

from espnet2.bin.enh_inference import SeparateSpeech  # noqa: E402

# USES runs a recurrent block over the whole recording rather than over a
# fixed window, so the cost is linear in the input and half a minute is what
# comfortably fits the GPU slice asked for below. Longer audio is trimmed
# rather than refused.
MAX_SECS = 30
GPU_SECONDS = 120
MODEL_TAG = os.environ.get(
    "ENH_MODEL_TAG", "espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw"
)
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

KEEP = "Same as the file"
EXAMPLE_WAV = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)


def _categories(model):
    """The training conditions a checkpoint names, in a stable order.

    ESPnet stores them as {index: name} so that a category id in the data
    means the same thing at inference; a checkpoint trained without them has
    none, and every menu below is then empty.
    """
    categories = getattr(model, "categories", None) or {}
    if isinstance(categories, dict):
        return [categories[key] for key in sorted(categories)]
    return list(categories)


def _rates(categories):
    """The sample rates a category list names, highest first.

    A category reads like "2ch_16k" or "1ch_16000Hz": two channels at
    16 kHz, one channel at 16 kHz. Recipes spell the rate both ways - USES
    one way, the URGENT 2025 recipe the other - and a checkpoint reached
    through ENH_MODEL_TAG comes from whichever recipe trained it, so both
    spellings fill the menu.
    """
    found = set()
    for category in categories:
        match = re.search(r"(\d+)k(?:Hz)?\b", category)
        if match:
            found.add(int(match.group(1)) * 1000)
            continue
        match = re.search(r"(\d+)\s*Hz", category)
        if match:
            found.add(int(match.group(1)))
    return sorted(found, reverse=True)


def _channels(categories):
    """The largest channel count a category list names, at least one."""
    found = {1}
    for category in categories:
        match = re.match(r"(\d+)ch", category)
        if match:
            found.add(int(match.group(1)))
    return max(found)


TITLE = "Universal speech enhancement"
DESCRIPTION = """# Universal speech enhancement

USES is a single speech enhancement model for every condition the task
usually splits into separate ones: noise, reverberation, one microphone or
several, and any sample rate. It was trained on the
[`egs2/universal_se_v1`](https://github.com/espnet/espnet/tree/master/egs2/universal_se_v1)
data with [ESPnet](https://github.com/espnet/espnet), and it is the model
`espnet enhance` uses by default.

Upload something noisy. Multi-channel files keep their channels on the way
in, because a model that can use them should be given them; what comes back
is one channel.
"""
ARTICLE = """Model:
[`espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw`](https://huggingface.co/espnet/Wangyou_Zhang_universal_train_enh_uses_refch0_2mem_raw).
Source of this Space:
[`egs2/universal_se_v1/enh1/demo`](https://github.com/espnet/espnet/tree/master/egs2/universal_se_v1/enh1/demo).

```bibtex
@inproceedings{zhang2023uses,
  title={Toward Universal Speech Enhancement for Diverse Input Conditions},
  author={Zhang, Wangyou and Saijo, Kohei and Wang, Zhong-Qiu and
          Watanabe, Shinji and Qian, Yanmin},
  booktitle={Proc. ASRU},
  year={2023}
}
```"""


enh = SeparateSpeech.from_pretrained(MODEL_TAG, device=DEVICE)

# The menu comes from the checkpoint. USES is sampling-frequency independent,
# so any rate runs; the rates it was trained on are the ones worth offering,
# and the recipe records them in the category names. The same names say how
# many channels it has ever seen, which is what the warning below compares
# against.
CATEGORIES = _categories(enh.enh_model)
RATES = _rates(CATEGORIES)
MAX_CHANNELS = _channels(CATEGORIES)
RATE_LABELS = [f"{rate // 1000} kHz" for rate in RATES]
RATE_OF_LABEL = dict(zip(RATE_LABELS, RATES))
# One output per speaker the checkpoint separates: this one enhances, so it
# has a single output, and a separation checkpoint reached through
# ENH_MODEL_TAG grows the column instead of dropping speakers on the floor.
OUTPUT_LABELS = (
    ["Enhanced"]
    if enh.num_spk == 1
    else [f"Speaker {i}" for i in range(1, enh.num_spk + 1)]
)


def _read(path):
    """The recording as (samples, channels) or (samples,), at its own rate."""
    if path is None:
        raise gr.Error("Record or upload some audio first.")
    # The header says how long the file is, so only the part that is enhanced
    # is decoded: an hour of audio uploaded to a 30 s demo is not read in full
    # first, and the warning is still about the file rather than about what
    # came back.
    if librosa.get_duration(path=path) > MAX_SECS:
        gr.Warning(
            f"Only the first {MAX_SECS} s were enhanced. "
            "Run the app yourself for longer audio - the model has no such "
            "limit."
        )
    # sr=None keeps the file's rate: the model is free to work at it, and
    # resampling is the user's choice below rather than a silent one here.
    speech, rate = librosa.load(path, sr=None, mono=False, duration=MAX_SECS)
    if speech.ndim > 1:  # librosa hands back (channels, samples)
        speech = speech.T
    channels = speech.shape[1] if speech.ndim > 1 else 1
    if channels > MAX_CHANNELS:
        gr.Warning(
            f"This file has {channels} channels and the model was trained on "
            f"at most {MAX_CHANNELS}; it will run, but off what it knows."
        )
    return speech, rate


@spaces.GPU(duration=GPU_SECONDS)
def predict(audio_path, rate_label):
    speech, rate = _read(audio_path)
    target = RATE_OF_LABEL.get(rate_label, rate)
    if target != rate:
        # axis=0 is the sample axis for both shapes _read returns
        speech = librosa.resample(speech, orig_sr=rate, target_sr=target, axis=0)
        rate = target
    waves = enh(speech[None, ...], fs=rate)
    return [(rate, wave[0]) for wave in waves]


EXAMPLES = [[EXAMPLE_WAV, KEEP]]


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Noisy audio"
            )
            rate = gr.Dropdown(
                [KEEP] + RATE_LABELS,
                value=KEEP,
                label="Process at",
                info="The rates this model was trained on",
            )
            button = gr.Button("Enhance", variant="primary")
        with gr.Column():
            outputs = [gr.Audio(label=label) for label in OUTPUT_LABELS]
    button.click(predict, [audio, rate], outputs)
    gr.Examples(
        EXAMPLES,
        inputs=[audio, rate],
        outputs=outputs,
        fn=predict,
        cache_examples=False,
    )
    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch()
