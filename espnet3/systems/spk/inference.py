"""Trial scoring entrypoint for trained speaker verification models."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from typeguard import typechecked

from espnet3.systems.spk.task import SpeakerTask


class Speech2Score:
    """Score verification trials with a trained ESPnet3 speaker model.

    The recipe feeds the two utterances of a trial, already cut into ``num_eval``
    crops by :class:`espnet2.train.preprocessor.SpkPreprocessor`, and gets back a
    single similarity score. Scoring averages the cosine similarity over every
    crop pair, matching the validation-time behaviour of
    :class:`espnet3.systems.spk.espnet_model.ESPnetSpeakerVerificationModel`.

    Args:
        train_config: Path to the ``config.yaml`` written by the ``train`` stage.
        model_file: Path to the model weights to score with.
        device: Torch device string.
        dtype: Floating point type used for the model input.

    Examples:
        >>> scorer = Speech2Score("exp/train/config.yaml", "exp/train/model.pth")
        >>> score = scorer(enrollment_crops, test_crops)
    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        """Load the speaker model used to embed and score trial utterances."""
        model, train_args = SpeakerTask.build_model_from_file(
            train_config, model_file, device
        )
        self.model = model.eval()
        self.train_args = train_args
        self.device = device
        self.dtype = dtype

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
