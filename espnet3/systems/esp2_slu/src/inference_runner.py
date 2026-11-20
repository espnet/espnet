"""Runners turning SLU and first-pass ASR hypotheses into contracted fields."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.esp2_slu.src.slu_output import split_intent


def _waveform(value):
    """Return the samples of an ``Audio``, or the value unchanged."""
    if not isinstance(value, Audio):
        return value
    return value.array.T if value.array.ndim == 2 else value.array


class SLUInferenceRunner(InferenceRunner):
    """Split a two-pass hypothesis into its intent and its transcript.

    ``SLUTask`` predicts ``"<intent> <transcript>"`` as one sequence, so the
    two scored quantities are both read off the single hypothesis: the intent
    for classification accuracy, the rest for WER and CER. Scoring the raw
    hypothesis against a reference transcript would charge the intent label as
    a word error.

    The backend decodes one utterance at a time, so a batched call is refused
    rather than silently scored on the wrong shape.
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand ``Speech2Understand`` the waveform it expects.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                ``Audio`` and ``transcript`` the text the post-decoder reads.

        Returns:
            A copy of ``inputs`` whose ``speech`` is a plain array.
        """
        inputs = dict(inputs)
        inputs["speech"] = _waveform(inputs["speech"])
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Format one hypothesis as intent, transcript and the raw sequence.

        Args:
            result: The n-best results of ``Speech2Understand``.
            batched: Whether the call was a batch.

        Returns:
            ``{"hyp": ..., "hyp_intent": ..., "hyp_transcript": ...}``.

        Raises:
            ValueError: If called on a batch.

        Examples:
            >>> result = [("news_query read the news", ["n"], [1], None)]
            >>> SLUInferenceRunner.format_output(result, batched=False)
            {'hyp': 'news_query read the news', 'hyp_intent': 'news_query', \
'hyp_transcript': 'read the news'}
        """
        if batched:
            raise ValueError(
                "The SLU backend decodes one utterance at a time. "
                "Set `batch_size: null` in the inference config."
            )
        hyp = result[0][0] or ""
        intent, transcript = split_intent(hyp)
        return {"hyp": hyp, "hyp_intent": intent, "hyp_transcript": transcript}


class FirstPassTranscriptRunner(InferenceRunner):
    """Keep a first-pass ASR hypothesis whole as the transcript.

    The first pass of this system is an ordinary ASR model: its hypothesis is
    words from the first token on. Splitting it the way
    :class:`SLUInferenceRunner` splits a two-pass hypothesis would throw the
    first word away as though it were an intent label.

    It predicts no intent, but the system's contract declares ``hyp_intent``
    and an overlay cannot drop an inherited output -- the contract rejects a
    null entry, and outputs may not be optional. So the field is written
    empty, which says "this mode has no intent" rather than inventing one.
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand ``Speech2Text`` the waveform it expects."""
        inputs = dict(inputs)
        inputs["speech"] = _waveform(inputs["speech"])
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Return the best hypothesis as the whole transcript.

        Args:
            result: The n-best results of ``Speech2Text`` (a list of them for
                a batch).
            batched: Whether the call was a batch.

        Returns:
            ``{"hyp": ..., "hyp_intent": "", "hyp_transcript": ...}``, or a
            list of them.

        Examples:
            >>> result = [("read the news", ["r"], [1], None)]
            >>> FirstPassTranscriptRunner.format_output(result, batched=False)
            {'hyp': 'read the news', 'hyp_intent': '', \
'hyp_transcript': 'read the news'}
        """

        def one(hypotheses):
            text = hypotheses[0][0] or ""
            return {"hyp": text, "hyp_intent": "", "hyp_transcript": text}

        return [one(r) for r in result] if batched else one(result)
