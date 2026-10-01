"""Decoding helpers, re-exported so conf/inference.yaml can say ``src.``.

The logic is the same for every OWSM recipe -- prompting the decoder from each
utterance's tags -- so it lives in the template rather than being copied here.
"""

from egs3.TEMPLATE.owsm.src.inference import (
    Speech2TextOWSM,
    build_output,
    prompt_of,
)

__all__ = ["Speech2TextOWSM", "build_output", "prompt_of"]
