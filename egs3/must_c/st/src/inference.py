"""Inference model wrapper and output formatting for the MuST-C en-de ST recipe.

``conf/inference.yaml`` uses both halves of this module::

    model:
      _target_: src.inference.Speech2TextShortSafe
    output_fn: src.inference.build_output

ESPnet3 decodes one sample at a time
(``espnet3/systems/base/inference_runner.py``: ``data = dataset[idx]``) and
writes one SCP file per key ``build_output`` returns, under
``${inference_dir}/<test_name>/`` -- so the keys below become ``hyp.scp``,
``ref.scp`` and ``src.scp``, which ``conf/metrics.yaml`` scores.
"""

from espnet2.bin.st_inference import Speech2Text
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)


class Speech2TextShortSafe(Speech2Text):
    """``Speech2Text`` that survives utterances too short to subsample.

    egs2 does not apply the 0.1-20 s filter to test sets (``st.sh``: "NOT
    applying to test_sets to keep original data"), so tst-COMMON contains 54 of
    2,641 utterances below the 7 frames a conv2d-subsampling conformer needs --
    the shortest is 0.050 s. Decoding one raises ``TooShortUttError``.

    espnet2 already handles this, but in its CLI loop
    (``espnet2/bin/st_inference.py``), not inside ``Speech2Text.__call__``;
    espnet3 calls the class directly, so a single 0.05 s utterance otherwise
    aborts the whole test set.

    The fallback mirrors egs2 exactly: emit ``" "`` and keep the utterance, so
    its reference is still scored (as a miss) rather than dropped -- dropping
    would inflate BLEU by removing the hardest items from the denominator.
    """

    def __call__(self, *args, **kwargs):
        """Decode, falling back to a blank hypothesis when subsampling fails."""
        try:
            return super().__call__(*args, **kwargs)
        except TooShortUttError:
            hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
            return [(" ", ["<space>"], [2], hyp)]


def build_output(data, model_output, idx):
    """Turn one decoding result into the dict ESPnet3 writes to SCP.

    Args:
        data: The raw dataset sample. ``conf/inference.yaml``'s test entries set
            ``return_utt_id: true``, so this carries a real MuST-C id.
        model_output: n-best list; ``[0][0]`` is the best hypothesis text.
        idx: Index of the sample within its test set.

    Returns:
        dict with ``utt_id``, ``hyp``, ``ref`` and ``src``.

    The training path cannot carry ``utt_id``: ``CommonCollateFn`` pads and
    stacks every value in the sample dict, so a str dies with
    ``AttributeError: 'str' object has no attribute 'dtype'`` on the first
    batch. Inference does not collate and requires an identifier
    (``InferenceRunner.idx_key`` defaults to ``"utt_id"``), so the Dataset emits
    it for test entries only. The ``str(idx)`` fallback keeps this working if
    that flag is off, as spgispeech and librispeech_100 do unconditionally --
    but integer keys make a bad hypothesis impossible to trace back to a talk.
    """
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": model_output[0][0],
        # `text` is the ST target (test entries use task="st"); `src_text` is
        # the English side, written so hypotheses can be read against input.
        "ref": data.get("text", ""),
        "src": data.get("src_text", ""),
    }
