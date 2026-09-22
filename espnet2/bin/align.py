#!/usr/bin/env python3
"""Line text up with the audio it was said in.

One algorithm - a forced alignment over a model's CTC head - for every
checkpoint that has one, whether it was trained as ASR or as speech-to-text::

    from espnet2.bin.align import ForcedAligner

    aligner = ForcedAligner.from_pretrained("espnet/owsm_ctc_v4_1B")
    for segment in aligner("audio.wav", ["the sale of the hotels",
                                         "is part of holiday's strategy"]):
        print(segment.start, segment.end, segment.score, segment.text)

Each segment also carries the tokens it was made of, with a time and a
probability each, which is where a word-level timestamp comes from.

The four alignment modules in espnet2.bin, and what each is::

    align.py        forced alignment over any CTC head; what `espnet align`
                    and the MCP server run
    asr_align.py    CTC segmentation with an ASR model, and its script
    s2t_align.py    CTC segmentation with an OWSM-CTC model, and its script
    ctc_segment.py  the CTC segmentation algorithm those two share; not an
                    entry point

This is one implementation rather than one per model: the algorithm needs
the CTC posteriors and the token ids, and nothing else about the model. The
`ctc_segmentation` package's algorithm stays because it is a different one,
with different output - a confidence score per utterance, from a search over
a window - and it is what the recipes here run, `egs2/owsm_v4/s2t1` among
them, for alignment-score data cleaning.

Measured against `ctc_segmentation` on the same posteriors, on
test_utils/ctc_align_test.wav: the ends agree within 0.03 s, the starts differ
because a forced alignment marks where a token is rather than partitioning the
timeline, and when the text does not cover the whole recording this returns
the same times it returns when it does.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from espnet2.utils.pretrained import ModelTagError, download_pretrained


@dataclass
class Token:
    """One token of the text, and where the model heard it."""

    text: str
    start: float
    end: float
    score: float


@dataclass
class Segment:
    """One utterance, and where the model heard it.

    `score` is the mean probability of its tokens: 1.0 is a perfect match,
    and a caption that does not belong to the audio scores near zero.

    It is a probability under this model, so the text has to be written the
    way the model writes it. On test_utils/ctc_align_test.wav with
    espnet/owsm_ctc_v4_1B, "The sale of the hotels" scores 0.99; the same
    words in capitals, as that recording's reference transcript has them,
    score 0.0000, because the vocabulary has no capitalised words and each
    one breaks into single letters. The times stay roughly right either way -
    it is the score that stops meaning anything.
    """

    text: str
    start: float
    end: float
    score: float
    tokens: List[Token]


class ForcedAligner:
    """Align utterances to a recording, on a published model's CTC head."""

    def __init__(self, model: Any = None, device: str = "cpu", **artifacts):
        """Wrap a loaded inference object, or build one from a model's files.

        Args:
            model: `espnet2.bin.s2t_inference.Speech2Text` or
                `espnet2.bin.asr_inference.Speech2Text`, already loaded. It
                is used for its encoder, its CTC head and its tokenizer, not
                for decoding.
            device: Where to load, when the files are given rather than a
                model.
            **artifacts: What a published model arrives with -
                `s2t_train_config` and `s2t_model_file`, or the `asr_` pair -
                as `from_pretrained` hands them over. Which of the two is
                present is what says how to load it.
        """
        if model is None:
            model = self._load(artifacts, device)
        self.model = model
        self.s2t = hasattr(model, "s2t_model")
        self.network = model.s2t_model if self.s2t else model.asr_model
        if getattr(self.network, "ctc", None) is None:
            raise ValueError(
                "this model has no CTC head, so there is nothing to align on"
            )

    @staticmethod
    def _load(artifacts: dict, device: str):
        """The inference class the artifacts were published for."""
        if "s2t_train_config" in artifacts:
            from espnet2.bin.s2t_inference import Speech2Text
        elif "asr_train_config" in artifacts:
            from espnet2.bin.asr_inference import Speech2Text
        else:
            raise ModelTagError(
                f"these files are not a model `espnet align` can read: "
                f"{sorted(artifacts)} is neither an S2T nor an ASR config. "
                f"`espnet models` names the default."
            )
        return Speech2Text(**artifacts, device=device)

    @classmethod
    def from_pretrained(cls, model_tag: Optional[str] = None, **kwargs):
        """Build from a tag, loading whichever inference class fits it.

        A published model says which it is: it arrives with `s2t_train_config`
        or with `asr_train_config`, and the constructor reads that. A tag for
        neither is named rather than half-loaded.
        """
        if model_tag is not None:
            kwargs.update(download_pretrained(model_tag))
        return cls(**kwargs)

    def _ids(self, text: str) -> List[int]:
        """The token ids of one utterance, without blanks."""
        tokens = self.model.tokenizer.text2tokens(text)
        ids = self.model.converter.tokens2ids(tokens)
        return [i for i in ids if i != self.network.blank_id]

    def _warn_about_spelling(self, utterance: str, ids: List[int]) -> None:
        """Say so when the vocabulary has a far shorter spelling of this text.

        A score is a probability under this model, so text spelled a way the
        model never saw gets aligned but not usefully scored - the times stay
        roughly right, which is what makes it confusing. On
        test_utils/ctc_align_test.wav with espnet/owsm_ctc_v4_1B, "THE SALE
        OF THE HOTELS" is 18 tokens and scores 0.0000, where "The sale of the
        hotels" is 6 and scores 0.99.

        Which case a model wants is the model's own business - one trained on
        WSJ wants capitals - so this compares instead of assuming: if the same
        words in another case need less than half as many tokens, that is the
        spelling this vocabulary has, and the caller is told which.
        """
        others = {utterance.lower(), utterance.upper()} - {utterance}
        for other in sorted(others):
            try:
                shorter = self._ids(other)
            except Exception:  # noqa: BLE001 - a diagnostic never breaks a run
                # a tokenizer or converter that cannot encode the variant at
                # all says nothing about the text that was given
                continue
            if shorter and len(shorter) * 2 <= len(ids):
                warnings.warn(
                    f"{utterance!r} is {len(ids)} tokens for this model, "
                    f"where {other!r} is {len(shorter)}: the text is spelled "
                    f"a way the vocabulary does not have, and the score will "
                    f"be near zero however well the audio matches",
                    UserWarning,
                    stacklevel=3,
                )
                return

    @torch.no_grad()
    def log_probs(self, speech: np.ndarray) -> np.ndarray:
        """CTC log posteriors for a recording, as (frames, vocabulary)."""
        if self.s2t:
            return self.model.ctc_log_probs(speech)
        speech_t = (
            torch.tensor(speech).unsqueeze(0).to(getattr(torch, self.model.dtype))
        )
        lengths = speech_t.new_full([1], dtype=torch.long, fill_value=speech_t.size(1))
        enc, _ = self.network.encode(
            **{
                "speech": speech_t.to(self.model.device),
                "speech_lengths": lengths.to(self.model.device),
            }
        )
        if isinstance(enc, tuple):
            enc = enc[0]
        return self.network.ctc.log_softmax(enc)[0].cpu().numpy()

    def __call__(
        self,
        speech: Union[str, Path, torch.Tensor, np.ndarray],
        utterances: Sequence[str],
        fs: Optional[int] = None,
    ) -> List[Segment]:
        """Align each utterance to the recording, in the order given.

        Args:
            speech: A path, or audio at the model's sample rate.
            utterances: What was said, one string per utterance.
            fs: Unused; accepted so that a caller does not have to know that
                a path is resampled for it.

        Returns:
            One Segment per utterance, in the order given.
        """
        import torchaudio

        if not list(utterances):
            raise ValueError("give the utterances to align")
        audio = self._read(speech)
        emissions = self.log_probs(audio)
        frames_per_sec = len(emissions) / (len(audio) / self._sample_rate())

        ids, spans = [], []
        for utterance in utterances:
            piece = self._ids(utterance)
            if not piece:
                raise ValueError(f"nothing to align in {utterance!r}")
            self._warn_about_spelling(utterance, piece)
            spans.append((len(ids), len(ids) + len(piece)))
            ids.extend(piece)
        # CTC needs a blank frame between two of the same token in a row, so
        # what has to fit is the tokens plus those blanks. Without counting
        # them, a text that is a few frames too long got torchaudio's
        # "targets length is too long for CTC" instead of this sentence.
        repeats = sum(1 for first, second in zip(ids, ids[1:]) if first == second)
        if len(ids) + repeats > len(emissions):
            needed = f"{len(ids)} tokens"
            if repeats:
                needed += f" and {repeats} blank(s) between repeated ones"
            raise ValueError(
                f"{needed} to align against {len(emissions)} frames: "
                f"this text cannot fit in this recording"
            )

        labels, scores = torchaudio.functional.forced_align(
            torch.tensor(emissions).unsqueeze(0),
            torch.tensor([ids], dtype=torch.int32),
            torch.tensor([len(emissions)]),
            torch.tensor([len(ids)]),
            blank=self.network.blank_id,
        )
        merged = torchaudio.functional.merge_tokens(labels[0], scores[0].exp())

        segments = []
        for (start, end), utterance in zip(spans, utterances):
            pieces = merged[start:end]
            tokens = [
                Token(
                    text=self.network.token_list[piece.token],
                    start=piece.start / frames_per_sec,
                    end=piece.end / frames_per_sec,
                    score=float(piece.score),
                )
                for piece in pieces
            ]
            segments.append(
                Segment(
                    text=utterance,
                    start=tokens[0].start,
                    end=tokens[-1].end,
                    score=float(np.mean([t.score for t in tokens])),
                    tokens=tokens,
                )
            )
        return segments

    @property
    def sample_rate(self) -> int:
        """The rate this model's audio has to arrive at.

        Public because a caller that reads its own audio - a Space drawing a
        waveform, a script batching files - has to resample to the same rate
        this class would, and hard-coding 16000 makes a checkpoint at another
        rate quietly wrong.
        """
        return self._sample_rate()

    def _sample_rate(self) -> int:
        rate = getattr(self.model, "sample_rate", None)
        if rate:
            return int(rate)
        conf = getattr(self.model, "preprocessor_conf", None) or {}
        return int(conf.get("fs", 16000))

    def _read(self, speech) -> np.ndarray:
        if hasattr(self.model, "read_audio"):
            return self.model.read_audio(speech)
        if isinstance(speech, (str, Path)):
            import librosa

            audio, _ = librosa.load(str(speech), sr=self._sample_rate())
            return audio
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()
        speech = np.asarray(speech, dtype=np.float32)
        return speech.mean(axis=1) if speech.ndim == 2 else speech
