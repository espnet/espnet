"""OWSM v4: speech recognition, translation and language ID, with a search.

Published as https://huggingface.co/spaces/espnet/owsm-v4; the source lives in
espnet, at egs2/owsm_v4/s2t1/demo. Its encoder-only sibling is
egs2/owsm_ctc_v4/s2t1/demo, and the two offer the same tasks so that the
models can be compared on the same audio.

The page is not here. It is built by espnet2.bin.demo - the module `espnet
demo` serves - from the checkpoint this file loads, which is also where the
text prompt comes from: this model has a decoder, so the page offers a box to
prime it with. This app is the loading, the device and the ZeroGPU slice.
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

import torch  # noqa: E402

from espnet2.bin.demo import build_app  # noqa: E402
from espnet2.bin.s2t_inference import Speech2Text  # noqa: E402

MODEL_TAG = os.environ.get("OWSM_MODEL_TAG", "espnet/owsm_v4_medium_1B")
# ZeroGPU gives a decorated call a fixed slice of GPU time and kills it at the
# end, so the demo asks for a slice and refuses audio it could not finish in
# one. The page refuses audio longer than two minutes, which is the same
# number in espnet2.bin.demo.
GPU_SECONDS = 120
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
demo = build_app(
    s2t,
    device=DEVICE,
    model_tag=MODEL_TAG,
    # the Run button is the only thing that touches the GPU
    wrap=spaces.GPU(duration=GPU_SECONDS),
    title="OWSM v4",
    description="""# OWSM v4

[OWSM](https://www.wavlab.org/activities/2024/owsm/) is the Open
Whisper-style Speech Model from [CMU WAVLab](https://www.wavlab.org/),
trained with [ESPnet](https://github.com/espnet/espnet). This is the
encoder-decoder one: it searches, which is slower and is what lets a text
prompt steer the answer.

Its encoder-only sibling runs the same tasks in one pass per window:
[espnet/owsm-ctc-v4](https://huggingface.co/spaces/espnet/owsm-ctc-v4).
""",
)


if __name__ == "__main__":
    demo.launch()
