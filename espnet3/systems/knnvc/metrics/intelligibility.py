"""Intelligibility of converted speech, scored with an ASR model.

kNN-VC is evaluated on how well the *content* of the source utterance survives
conversion: the converted waveform is transcribed and compared against the
source transcript. The authors report 6.29% WER and 2.34% CER on LibriSpeech
dev-clean for the prematched vocoder (https://github.com/bshall/knn-vc).

The ``infer`` stage already writes everything this needs: ``wav.scp`` points at
the converted audio and ``ref.scp`` carries the source transcript.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class ASRIntelligibility(BaseMetric):
    """Transcribe converted audio and score it against the source transcript.

    Args:
        asr: Either a model tag understood by
            ``espnet2.bin.asr_inference.Speech2Text.from_pretrained``, or an
            already-built object exposing the same ``__call__(speech)`` ->
            ``[(text, ...), ...]`` interface. There is no default: which ASR
            scores the conversions is a recipe's choice, and passing a built
            object lets a recipe score with any ASR it likes.
        wav_key: Metric input alias holding the converted-audio SCP.
        ref_key: Metric input alias holding the reference transcript SCP.
        clean_types: Optional cleaner types passed to ``TextCleaner``, applied
            to both sides before scoring.
        device: Device the ASR model runs on when built from a tag.
        sample_rate: Sample rate the ASR model expects.

    Example:
        .. code-block:: yaml

            metrics:
              - metric:
                  _target_: espnet3.systems.knnvc.metrics.intelligibility.ASRIntelligibility  # noqa: E501
                  asr: espnet/some_librispeech_asr_model
                  device: cuda
                inputs:
                  wav: wav
                  ref: ref
    """

    def __init__(
        self,
        asr: Any,
        wav_key: str = "wav",
        ref_key: str = "ref",
        clean_types: Optional[Iterable[str]] = None,
        device: str = "cpu",
        sample_rate: int = 16000,
    ) -> None:
        """Store the scoring configuration; the ASR model is built lazily."""
        self.asr = asr
        self.wav_key = wav_key
        self.ref_key = ref_key
        self.cleaner = TextCleaner(clean_types)
        self.device = device
        self.sample_rate = int(sample_rate)
        self._speech2text = None if isinstance(asr, str) else asr

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

    def build_asr(self):
        """Return the ASR model, building it from the tag on first use.

        Returns:
            The object used to transcribe, built once and reused.
        """
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

    def _read(self, path: str) -> np.ndarray:
        """Read one converted waveform as float32 mono.

        Args:
            path: Path to a WAV file written by the ``infer`` stage.

        Returns:
            Waveform array.

        Raises:
            ValueError: If the file's sample rate is not ``sample_rate``.
        """
        speech, rate = sf.read(path, dtype="float32")
        if rate != self.sample_rate:
            raise ValueError(
                f"{path} is {rate} Hz but the ASR model expects {self.sample_rate} Hz"
            )
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        return np.ascontiguousarray(speech)

    def transcribe(self, path: str) -> str:
        """Transcribe one converted utterance.

        Args:
            path: Path to the WAV file.

        Returns:
            The best hypothesis, or an empty string if the model returned none.
        """
        results = self.build_asr()(self._read(path))
        if not results:
            return ""
        return results[0][0]

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Transcribe every converted utterance and score it.

        Args:
            data: Mapping of metric input aliases to SCP paths; needs
                ``wav_key`` (converted audio) and ``ref_key`` (transcript).
            test_name: Test set name, used for the alignment output directory.
            inference_dir: Directory the alignment file is written under.

        Returns:
            ``{"ASR_WER": <percentage>, "ASR_CER": <percentage>}``.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        self._ensure_jiwer()
        refs: List[str] = []
        hyps: List[str] = []
        for _, row in self.iter_inputs(data, self.ref_key, self.wav_key):
            refs.append(self._clean(row[self.ref_key]))
            hyps.append(self._clean(self.transcribe(row[self.wav_key])))

        word_details = jiwer.process_words(refs, hyps)
        char_details = jiwer.process_characters(refs, hyps)

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "asr_wer_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(word_details))

        return {
            "ASR_WER": round(word_details.wer * 100, 2),
            "ASR_CER": round(char_details.cer * 100, 2),
        }
