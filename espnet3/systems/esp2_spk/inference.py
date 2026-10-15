"""The speaker verification system's side of :mod:`espnet3.api.inference`.

A trial is two utterances; the model answers how likely they are to share a
speaker, as one similarity score. :class:`Inference` takes the two
waveforms, cuts each into evenly spaced fixed-length crops exactly as
validation does during training, and returns ``{"score": float}``. The
model itself, and the crops, are :class:`ESPnet2Speech2Score`'s.

Examples:
    From a bundle, by directory or Hub tag::

        >>> from espnet3.systems.esp2_spk.inference import Inference
        >>> model = Inference.from_pretrained("exp/train/model_pack")
        >>> model("enroll.wav", "test.wav")
        {'score': 0.71}

    In the ``infer`` stage, ``inference.yaml`` names this class with the
    scorer's own arguments::

        model:
          _target_: espnet3.systems.esp2_spk.inference.Inference
          train_config: ${exp_dir}/config.yaml
          model_file: ${exp_dir}/valid.eer.ave_3best.pth
          num_eval: 10
"""

from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import torch
from typeguard import typechecked

from espnet3.api.inference import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference
from espnet3.systems.esp2_spk.task import SpeakerTask


def eval_crops(audio: np.ndarray, crop_samples: int, num_eval: int) -> np.ndarray:
    """Cut one utterance into ``num_eval`` evenly spaced crops.

    The same crops :class:`espnet2.train.preprocessor.SpkPreprocessor` takes
    with ``train=False``, so a score here matches the validation EER: an
    utterance shorter than one crop is first wrapped around to fill it.

    Args:
        audio: 1-D waveform.
        crop_samples: Length of one crop, in samples.
        num_eval: Number of crops.

    Returns:
        ``(num_eval, crop_samples)`` array.

    Examples:
        >>> eval_crops(np.arange(10.0), crop_samples=4, num_eval=3)[:, 0]
        array([0., 3., 6.])
    """
    if len(audio) < crop_samples:
        audio = np.pad(audio, (0, crop_samples - len(audio) + 1), "wrap")
    starts = np.linspace(0, len(audio) - crop_samples, num=num_eval)
    return np.stack([audio[int(s) : int(s) + crop_samples] for s in starts])


class ESPnet2Speech2Score:
    """Score verification trials with a checkpoint of the ESPnet2 speaker task.

    The recipe feeds the two utterances of a trial, already cut into ``num_eval``
    crops by :class:`espnet2.train.preprocessor.SpkPreprocessor`, and gets back a
    single similarity score. Scoring averages the cosine similarity over every
    crop pair, matching the validation-time behaviour of
    :class:`espnet3.systems.esp2_spk.src.espnet_model.ESPnetSpeakerVerificationModel`.

    The ``ESPnet2`` in the name is about the checkpoint format, not the trainer:
    :class:`espnet3.systems.esp2_spk.task.SpeakerTask` is a copy of the ESPnet2
    speaker task, so this loads a ``config.yaml`` written by either the ESPnet2
    ``spk`` recipes or this ESPnet3 one. It always rebuilds the model as an
    ``ESPnetSpeakerVerificationModel``, which is what adds the trial scoring the
    ESPnet2 model class does not have.

    Args:
        train_config: Path to the ``config.yaml`` written by the ``train`` stage.
        model_file: Path to the model weights to score with.
        device: Torch device string.
        dtype: Floating point type used for the model input.
        sample_rate: Rate the model was trained at, in Hz.
        target_duration: Length of one evaluation crop, in seconds.
        num_eval: Number of crops taken from each utterance.

    Examples:
        >>> scorer = ESPnet2Speech2Score("exp/train/config.yaml", "exp/train/model.pth")
        >>> score = scorer(enrollment_crops, test_crops)
        >>> score = scorer.score(enrollment_wav, test_wav)
    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        device: str = "cpu",
        dtype: str = "float32",
        sample_rate: int = 16000,
        target_duration: float = 3.0,
        num_eval: int = 10,
    ):
        """Load the speaker model used to embed and score trial utterances."""
        model, train_args = SpeakerTask.build_model_from_file(
            train_config, model_file, device
        )
        self.model = model.eval()
        self.train_args = train_args
        self.device = device
        self.dtype = dtype
        self.sample_rate = sample_rate
        self.crop_samples = int(target_duration * sample_rate)
        self.num_eval = num_eval

    def score(self, speech: np.ndarray, speech2: np.ndarray) -> float:
        """Return the similarity score of two whole 1-D waveforms."""
        return self(
            eval_crops(speech, self.crop_samples, self.num_eval),
            eval_crops(speech2, self.crop_samples, self.num_eval),
        )

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech2: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Return the similarity score of one trial pair.

        Args:
            speech: ``(num_eval, Samples)`` crops of the first utterance.
            speech2: ``(num_eval, Samples)`` crops of the second utterance.

        Returns:
            Similarity score, higher for utterances of the same speaker.
        """
        score = self.model.score_trials(self._to_batch(speech), self._to_batch(speech2))
        return float(score.squeeze())

    @torch.no_grad()
    def extract_embedding(self, speech: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Return the mean unit-norm speaker embedding of one utterance.

        Args:
            speech: ``(num_eval, Samples)`` crops of the utterance.

        Returns:
            Speaker embedding as a 1-D array.

        Examples:
            Enrol a speaker once and reuse the embedding, instead of paying for
            the forward pass on every trial:

            >>> scorer = ESPnet2Speech2Score(
            ...     "exp/train/config.yaml", "exp/train/valid.eer.ave_3best.pth"
            ... )
            >>> preprocessor = SpkPreprocessor(train=False, target_duration=3.0)
            >>> crops = preprocessor("utt1", {"speech": wav})["speech"]
            >>> enrolled = scorer.extract_embedding(crops)
            >>> enrolled.shape  # projector output size, 192 for RawNet3
            (192,)

            The embedding is the mean of the unit-norm crop embeddings, so a
            plain dot product against a second one reproduces :meth:`__call__`:

            >>> other = scorer.extract_embedding(other_crops)
            >>> float(enrolled @ other)  # == scorer(crops, other_crops)
            0.87
        """
        embd = self.model.extract_crop_embeddings(self._to_batch(speech))
        return embd.mean(dim=1).squeeze(0).cpu().numpy()

    def _to_batch(self, speech: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert one utterance into a single-item batch on the model device."""
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        speech = speech.to(getattr(torch, self.dtype))
        if speech.dim() == 1:
            speech = speech.unsqueeze(0)
        return speech.unsqueeze(0).to(self.device)


class Inference(BackendInference):
    """Score a verification trial with a speaker model.

    Built as every :class:`BackendInference` is: ``from_pretrained`` from a
    ``pack_model`` bundle, ``Inference(train_config=..., model_file=...)``
    from :class:`ESPnet2Speech2Score`'s own arguments, which is what
    ``inference.yaml`` does, or ``Inference(scorer)`` around one already
    built.

    Examples:
        >>> model = Inference.from_pretrained("exp/train/model_pack")
        >>> model("enroll.wav", "test.wav")
        {'score': 0.71}
        >>> model.sample_rate
        16000
    """

    backend_class = "espnet3.systems.esp2_spk.inference.ESPnet2Speech2Score"
    inputs = (
        Field("speech", "audio", "Enrollment speech"),
        Field("speech2", "audio", "Test speech"),
    )
    outputs = (Field("score", "number", "Similarity score"),)

    def run(self, speech: Audio, speech2: Audio) -> Mapping[str, Any]:
        """Return the trial's score; higher means more likely one speaker.

        Args:
            speech: The enrollment utterance, mono, at :attr:`sample_rate`.
            speech2: The test utterance, mono, at :attr:`sample_rate`.

        Returns:
            ``{"score": float}``: the cosine similarity averaged over every
            pair of crops.
        """
        return {"score": self.backend.score(speech.array, speech2.array)}
