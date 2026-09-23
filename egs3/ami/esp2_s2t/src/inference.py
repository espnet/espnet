"""Inference output helpers for the AMI SOT recipe.

The measure stage scores two different views of the same hypothesis. cpWER
compares words, so it reads text with the timestamps removed. DER compares
speaker activity in time, so it reads the sequence with its timestamps
intact. ``build_output`` writes both, for the hypothesis and the reference.

A recipe sample cannot carry its own utterance id (see
``dataset/dataset.py``'s module docstring), so ``build_output`` resolves one
through ``current_utt_id``, which reads it from the split whichever
``AmiSotDataset`` the framework actually built for this run was constructed
with. That is the only utterance-id source on the inference path: there is
no second, independently loaded copy of a split's ids here that could drift
from it.
"""

from __future__ import annotations

import re
from typing import Any, Dict

try:
    from egs3.ami.esp2_s2t.dataset.dataset import current_utt_id
except ImportError:
    # A packed bundle ships src/ and dataset/ at its root with no egs3 package
    # above them; InferenceModel puts the bundle root on sys.path, so the
    # top-level spelling resolves there. current_utt_id's state lives in
    # os.environ, so which copy answers does not matter.
    from dataset.dataset import current_utt_id
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer

MODEL_TYPE = "whisper_multilingual"

# Everything below is built on first use, not at import time. Importing this
# module must not need the corpus or the Whisper assets, so that a unit test can
# import it anywhere.
_CONVERTER = None
_TOKENIZER = None


def _tokenizers():
    """Return the (converter, tokenizer) pair, built once on first use.

    ``keep_special_tokens`` is what preserves the ``<|x.xx|>`` timestamps
    through ``ids2tokens``. The default converter drops them.
    """
    global _CONVERTER, _TOKENIZER
    if _CONVERTER is None:
        _CONVERTER = OpenAIWhisperTokenIDConverter(MODEL_TYPE, keep_special_tokens=True)
        _TOKENIZER = OpenAIWhisperTokenizer(MODEL_TYPE)
    return _CONVERTER, _TOKENIZER


_TIMESTAMP_RE = re.compile(r"<\|\d+\.\d+\|>")


def strip_timestamps(text: str) -> str:
    """Remove Whisper timestamp tokens and collapse whitespace.

    Args:
        text: A serialized hypothesis or reference, for example
            ``"<|0.00|> hello<|1.20|> <sc> <|0.58|> world<|2.40|>"``.

    Returns:
        The same text without ``<|x.xx|>`` tokens and with runs of whitespace
        collapsed to one space, for example ``"hello <sc> world"``.
    """
    return " ".join(_TIMESTAMP_RE.sub(" ", text).split())


def _render(model_output: Any) -> str:
    """Render one hypothesis back to text with its timestamps.

    Args:
        model_output: What ``Speech2Text.__call__`` returned. Its first entry
            is one hypothesis, whose third element is the token id sequence.

    Returns:
        The rendered hypothesis, separator still in its trained form.
    """
    converter, tokenizer = _tokenizers()
    _text, _tokens, token_int, *_rest = model_output[0]
    # The first two ids are the language and task symbols, which belong to the
    # prompt rather than to the hypothesis.
    token_int = list(token_int)[2:]
    return tokenizer.tokens2text(converter.ids2tokens(token_int))


def build_output(data: Dict[str, Any], model_output: Any, idx: int) -> Dict[str, str]:
    """Build the SCP rows for one utterance group.

    Args:
        data: The dataset sample, which holds the reference under ``text``.
        model_output: The value ``Speech2Text`` returned for this sample.
        idx: Position of this sample in ``wav.scp`` order.

    Returns:
        ``utt_id``, plus ``hyp``/``ref`` for cpWER and ``hyp_sot``/``ref_sot``
        for DER. Text cleaning is left to the metrics, so that the same files
        can be rescored under a different normalizer.

    Raises:
        RuntimeError: If no ``AmiSotDataset`` has been constructed yet in
            this process (see ``current_utt_id``), so there is no id to
            attach to ``idx``.
    """
    # Written with the symbol the model was trained on, not a scoring-side
    # spelling: the SCP files should say what the model actually emitted. The
    # metrics split on that same symbol, so nothing downstream needs a rewrite.
    hyp_sot = _render(model_output)
    ref_sot = str(data.get("text", ""))
    return {
        "utt_id": current_utt_id(idx),
        "hyp": strip_timestamps(hyp_sot),
        "ref": strip_timestamps(ref_sot),
        "hyp_sot": hyp_sot,
        "ref_sot": ref_sot,
    }
