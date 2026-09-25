"""POWSM-CTC: the phones in a recording, in IPA.

Published as https://huggingface.co/spaces/espnet/powsm-ctc; the source lives
in espnet, at egs2/powsm_ctc/s2t1/demo.

Unlike the demos beside it, this app has no interface of its own. The page is
built by `espnet2.bin.demo`, the module `espnet demo` serves, which reads the
menus, the decoding window and the tasks off the checkpoint it was handed: a
model with `<pr>` in its token list - a phonetic one - is offered phone
recognition beside transcription. So this file is the loading, the device and
the ZeroGPU slice, and nothing about the page.

That module is importable from a release as of 202610.post2, which
requirements.txt pins. The two OWSM apps still keep their own copies, made
before it existed; test/espnet2/bin/test_demo_apps.py holds those to it.
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

# POWSM-CTC: the CTC variant, which reads a recording of any length in one
# pass per window. Its encoder-decoder sibling repeats itself on the padding
# a short clip is given, which a browser demo cannot hide.
MODEL_TAG = os.environ.get("POWSM_MODEL_TAG", "espnet/powsm_ctc")
# ZeroGPU gives a decorated call a fixed slice of GPU time and kills it at the
# end. The page refuses audio longer than two minutes, so the slice matches.
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

s2t = Speech2Text.from_pretrained(MODEL_TAG, device=DEVICE)
demo = build_app(
    s2t,
    device=DEVICE,
    model_tag=MODEL_TAG,
    # the Run button is the only thing that touches the GPU
    wrap=spaces.GPU(duration=GPU_SECONDS),
    title="POWSM-CTC",
    description="""# POWSM-CTC

[POWSM](https://arxiv.org/abs/2510.24992) is a phonetic foundation model:
speech in, the phones that were said out, in IPA. This is the encoder-only
variant, trained on [IPAPack++](https://huggingface.co/anyspeech) with
[ESPnet](https://github.com/espnet/espnet) and released with
[PRiSM](https://arxiv.org/abs/2601.14046).

It is the model `espnet phonemize` loads. Its encoder-decoder sibling is
[espnet/powsm](https://huggingface.co/espnet/powsm).
""",
)


if __name__ == "__main__":
    demo.launch()
