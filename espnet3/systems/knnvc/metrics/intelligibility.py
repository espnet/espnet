"""Intelligibility of converted speech, scored with an ASR model.

kNN-VC is evaluated on how well the *content* of the source utterance survives
conversion: the converted waveform is transcribed and compared against the
source transcript. The authors report 6.29% WER and 2.34% CER on LibriSpeech
dev-clean for the prematched vocoder (https://github.com/bshall/knn-vc).

The ``infer`` stage writes ``wav.scp``, the converted audio; the transcript is
the test set's own ``text`` column, bound as ``dataset:text`` in the metrics
config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import soundfile as sf

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.contract.kinds.audio import Audio

logger = logging.getLogger(__name__)

_COUNTS = ("hits", "substitutions", "deletions", "insertions")


class ASRIntelligibility(BaseMetric):
    """Transcribe converted audio and score it against the source transcript.

    Declares ``ref`` (``text``) and ``wav`` (``audio``) as inputs; the
    metrics config binds them, ``ref`` to the test set's transcript column
    (``dataset:text``) and ``wav`` to the ``wav.scp`` of the ``infer`` stage.
    ``update`` gets the SCP values, so ``wav`` holds paths to the converted
    WAV files; an :class:`~espnet3.contract.kinds.audio.Audio` or a bare
    array at ``sample_rate`` is accepted too, for a caller with decoded
    audio in hand. Word and character edit counts are summed over every
    batch, and :meth:`compute` turns them into one corpus WER and CER.

    Args:
        asr: Either a model tag understood by
            ``espnet2.bin.asr_inference.Speech2Text.from_pretrained``, or an
            already-built object exposing the same ``__call__(speech)`` ->
            ``[(text, ...), ...]`` interface. There is no default: which ASR
            scores the conversions is a recipe's choice, and passing a built
            object lets a recipe score with any ASR it likes.
        clean_types: Optional cleaner types passed to ``TextCleaner``, applied
            to both sides before scoring.
        device: Device the ASR model runs on when built from a tag.
        sample_rate: Sample rate the ASR model expects; a WAV at another
            rate is refused.

    Example:
        .. code-block:: yaml

            metrics:
              - metric:
                  _target_: espnet3.systems.knnvc.metrics.intelligibility.ASRIntelligibility  # noqa: E501
                  asr: espnet/some_librispeech_asr_model
                  device: cuda
                inputs:
                  wav: wav
                  ref: dataset:text

        >>> metric = ASRIntelligibility(asr=built_speech2text)
        >>> metric.update(ref=["the cat sat"], wav=["converted/utt1.wav"])
        >>> metric.compute()
        {'ASR_WER': 0.0, 'ASR_CER': 0.0}
    """

    inputs = (Field("ref", "text"), Field("wav", "audio"))
    outputs = (Field("ASR_WER", "number"), Field("ASR_CER", "number"))

    def __init__(
        self,
        asr: Any,
        clean_types: Optional[Iterable[str]] = None,
        device: str = "cpu",
        sample_rate: int = 16000,
    ) -> None:
        """Store the scoring configuration; the ASR model is built lazily."""
        self.asr = asr
        self.cleaner = TextCleaner(clean_types)
        self.device = device
        self.sample_rate = int(sample_rate)
        self._speech2text = None if isinstance(asr, str) else asr
        super().__init__()
        for prefix in ("word_", "char_"):
            for name in _COUNTS:
                self.add_state(prefix + name, 0, "sum")
        self._alignments: List[str] = []

    def reset(self) -> None:
        """Reset the counts and drop alignment text from an unfinished run."""
        super().reset()
        self._alignments = []

    def _ensure_jiwer(self) -> None:
        """Raise a clear error if the optional jiwer dependency is missing.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute WER/CER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def _build_asr(self):
        """Return the ASR model, building it from the tag on first use."""
        if self._speech2text is None:
            from espnet2.bin.asr_inference import Speech2Text

            logger.info("Loading ASR model for intelligibility scoring: %s", self.asr)
            self._speech2text = Speech2Text.from_pretrained(
                self.asr, device=self.device
            )
        return self._speech2text

    def _clean(self, text: str) -> str:
        """Clean text, substituting a placeholder for an empty result."""
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def _samples(self, wav: Any) -> np.ndarray:
        """Return one converted utterance as float32 mono at ``sample_rate``.

        Args:
            wav: A path to a WAV file written by the ``infer`` stage, an
                ``Audio``, or a bare array taken to be at ``sample_rate``.

        Raises:
            ValueError: If a file's (or an ``Audio``'s) rate is not
                ``sample_rate``.
        """
        if isinstance(wav, (str, Path)):
            speech, rate = sf.read(str(wav), dtype="float32")
            if rate != self.sample_rate:
                raise ValueError(
                    f"{wav} is {rate} Hz but the ASR model expects "
                    f"{self.sample_rate} Hz"
                )
            if speech.ndim > 1:
                speech = speech.mean(axis=1)
            return np.ascontiguousarray(speech)
        if isinstance(wav, Audio):
            if wav.rate != self.sample_rate:
                raise ValueError(
                    f"audio is {wav.rate} Hz but the ASR model expects "
                    f"{self.sample_rate} Hz"
                )
            return wav.mono().array
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(wav, dtype=np.float32)).reshape(-1)

    def _transcribe(self, wav: Any) -> str:
        """Return the best hypothesis for one utterance, or ``""`` if none."""
        results = self._build_asr()(self._samples(wav))
        if not results:
            return ""
        return results[0][0]

    def update(self, *, ref: Sequence[str], wav: Sequence[Any]) -> None:
        """Transcribe one batch and add its word and character edit counts.

        Args:
            ref: Source transcripts, one per converted utterance.
            wav: The converted utterances (paths, ``Audio`` or arrays),
                aligned with ``ref``.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        self._ensure_jiwer()
        if not ref:
            return
        refs = [self._clean(text) for text in ref]
        hyps = [self._clean(self._transcribe(one)) for one in wav]
        word_out = jiwer.process_words(refs, hyps)
        char_out = jiwer.process_characters(refs, hyps)
        for prefix, out in (("word_", word_out), ("char_", char_out)):
            for name in _COUNTS:
                setattr(
                    self,
                    prefix + name,
                    getattr(self, prefix + name) + getattr(out, name),
                )
        if self.collect_artifacts:
            self._alignments.append(
                jiwer.visualize_alignment(word_out, show_measures=False)
            )

    def _rate(self, prefix: str) -> float:
        """Return the error rate of the ``prefix`` counts as a percentage."""
        hits, subs, dels, ins = (getattr(self, prefix + name) for name in _COUNTS)
        n = hits + subs + dels
        if n == 0:
            raise ValueError("nothing to score: the references have no words")
        return round(100 * (subs + dels + ins) / n, 2)

    def compute(self) -> Dict[str, float]:
        """Return the corpus WER and CER of everything added so far.

        Returns:
            ``{"ASR_WER": <percentage>, "ASR_CER": <percentage>}``.

        Raises:
            ValueError: If the references have no words.
        """
        return {"ASR_WER": self._rate("word_"), "ASR_CER": self._rate("char_")}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the word alignments of this run to ``test_dir/asr_wer_alignment``."""
        summary = (
            f"\nASR_WER: {self._rate('word_'):.2f} ASR_CER: {self._rate('char_'):.2f}\n"
        )
        (test_dir / "asr_wer_alignment").write_text(
            "".join(self._alignments) + summary, encoding="utf-8"
        )
        self._alignments = []
