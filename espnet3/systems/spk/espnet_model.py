"""Speaker model that also scores verification trials during validation."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetSpeakerVerificationModel(ESPnetSpeakerModel):
    """Speaker embedding model with open-set trial scoring.

    Training batches are handled exactly as in
    :class:`espnet2.spk.espnet_model.ESPnetSpeakerModel`: one utterance per
    sample with a closed-set speaker label. Validation batches instead carry a
    trial pair (``speech`` and ``speech2``, each already cut into ``num_eval``
    crops by :class:`espnet2.train.preprocessor.SpkPreprocessor`) together with
    a binary ``spk_labels``.

    A closed-set validation loss says little about verification performance, so
    trial batches skip the classification head. Their cosine similarities and
    labels accumulate in ``trial_scores`` / ``trial_labels`` until
    :class:`espnet3.systems.spk.callbacks.SpeakerVerificationScoring` reduces
    one epoch worth of them into EER and minDCF.

    Examples:
        >>> loss, stats, weight = model(speech=wavs, spk_labels=labels)
        >>> _ = model(speech=enroll, speech2=test, spk_labels=is_target)
        >>> len(model.trial_scores)
        1
    """

    def __init__(self, *args, **kwargs):
        """Initialize the model and the per-epoch trial buffers."""
        super().__init__(*args, **kwargs)
        self.trial_scores: List[torch.Tensor] = []
        self.trial_labels: List[torch.Tensor] = []

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
        spk_labels: Optional[torch.Tensor] = None,
        speech2: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run one training step, or score one batch of verification trials.

        Args:
            speech: ``(Batch, Samples)`` for training, or
                ``(Batch, num_eval, Samples)`` for trials.
            speech_lengths: Unpadded length of each ``speech`` entry. Ignored
                for trials, whose crops all have the same length.
            spk_labels: Speaker index for training, or a binary target/nontarget
                flag for trials.
            speech2: Second utterance of each trial. Its presence is what
                selects trial scoring over classification.
            **kwargs: Remaining fields produced by the collate function, such as
                ``task_tokens``.

        Returns:
            The usual ``(loss, stats, weight)`` triple. For trials the loss is a
            constant, because the epoch metric is computed by the scoring
            callback rather than per batch. ``weight`` is the batch size in
            both cases: the Lightning module passes it as the ``batch_size`` of
            the logged stats, so that partial batches do not skew an epoch
            average. Trial batches log no stats and would tolerate ``None``,
            but returning the batch size keeps one contract for both branches.

        Raises:
            ValueError: If a trial batch does not carry ``spk_labels``.

        Examples:
            Training. One utterance per sample, labelled with its speaker
            index, scored by the closed-set classification head:

            >>> loss, stats, weight = model(
            ...     speech=torch.randn(4, 48000),
            ...     speech_lengths=torch.full((4,), 48000),
            ...     spk_labels=torch.randint(0, 7205, (4, 1)),
            ... )
            >>> sorted(stats), int(weight)
            (['accuracy', 'loss'], 4)

            Validation. One trial pair per sample, labelled 1 for target and 0
            for nontarget. The similarity is buffered for the epoch metric
            instead of contributing to a loss:

            >>> _ = model.eval()
            >>> loss, stats, weight = model(
            ...     speech=torch.randn(4, 10, 48000),  # (Batch, num_eval, Samples)
            ...     speech2=torch.randn(4, 10, 48000),
            ...     spk_labels=torch.tensor([[1], [0], [1], [0]]),
            ... )
            >>> float(loss), stats, len(model.trial_scores)
            (0.0, {}, 1)
        """
        if speech2 is None:
            return super().forward(
                speech=speech,
                speech_lengths=speech_lengths,
                spk_labels=spk_labels,
                **kwargs,
            )

        if spk_labels is None:
            raise ValueError(
                "A trial batch must carry `spk_labels` holding 1 for target and "
                "0 for nontarget pairs."
            )

        scores = self.score_trials(speech, speech2)
        self.trial_scores.append(scores.detach())
        self.trial_labels.append(spk_labels.detach().flatten())

        stats: Dict[str, torch.Tensor] = {}
        loss = scores.new_zeros(())
        return force_gatherable((loss, stats, speech.shape[0]), scores.device)

    def score_trials(self, speech: torch.Tensor, speech2: torch.Tensor) -> torch.Tensor:
        """Return one similarity score per trial pair in the batch.

        Each utterance contributes ``num_eval`` crops, and the trial score is
        the cosine similarity averaged over every crop pair, which is the
        standard VoxCeleb scoring protocol.

        Args:
            speech: ``(Batch, num_eval, Samples)`` enrollment crops.
            speech2: ``(Batch, num_eval, Samples)`` test crops.

        Returns:
            ``(Batch,)`` tensor of similarity scores.

        Examples:
            >>> from espnet3.systems.spk.task import SpeakerTask
            >>> model, _ = SpeakerTask.build_model_from_file(
            ...     "exp/train/config.yaml", "exp/train/valid.eer.ave_3best.pth"
            ... )
            >>> enroll = torch.randn(2, 10, 48000)  # (Batch, num_eval, Samples)
            >>> test = torch.randn(2, 10, 48000)
            >>> _ = model.eval()
            >>> with torch.no_grad():
            ...     scores = model.score_trials(enroll, test)
            >>> scores.shape
            torch.Size([2])

            Scoring an utterance against itself gives 1.0, so this is a quick
            sanity check on a checkpoint:

            >>> with torch.no_grad():
            ...     float(model.score_trials(enroll[:1], enroll[:1]))
            1.0

            :class:`espnet3.systems.spk.inference.ESPnet2Speech2Score` wraps this for
            one trial at a time, and handles the batching and the device.
        """
        embd = self.extract_crop_embeddings(speech)
        embd2 = self.extract_crop_embeddings(speech2)
        return torch.einsum("bid,bjd->bij", embd, embd2).flatten(1).mean(dim=1)

    def extract_crop_embeddings(self, speech: torch.Tensor) -> torch.Tensor:
        """Embed every crop of every utterance and L2-normalize the result.

        Args:
            speech: ``(Batch, num_eval, Samples)``, or ``(Batch, Samples)`` when
                a single crop per utterance is used.

        Returns:
            ``(Batch, num_eval, Dim)`` tensor of unit-norm speaker embeddings.

        Examples:
            >>> _ = model.eval()
            >>> with torch.no_grad():
            ...     embd = model.extract_crop_embeddings(
            ...         torch.randn(2, 10, 48000)  # (Batch, num_eval, Samples)
            ...     )
            >>> embd.shape  # Dim is the projector output size, 192 for RawNet3
            torch.Size([2, 10, 192])
            >>> torch.allclose(embd.norm(dim=-1), torch.ones(2, 10), atol=1e-5)
            True

            Averaging over the crop axis gives one embedding per utterance,
            which is what :meth:`ESPnet2Speech2Score.extract_embedding` returns:

            >>> embd.mean(dim=1).shape
            torch.Size([2, 192])
        """
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        batch_size, num_crop = speech.shape[:2]

        crops = speech.flatten(0, 1)
        lengths = crops.new_full((crops.shape[0],), crops.shape[1], dtype=torch.long)
        embd = super().forward(speech=crops, speech_lengths=lengths, extract_embd=True)
        embd = F.normalize(embd, p=2, dim=-1)
        return embd.view(batch_size, num_crop, -1)

    def pop_trials(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the buffered trial scores and labels, and clear the buffers.

        Public because it is the hand-off point to the scoring callback:
        :class:`espnet3.systems.spk.callbacks.SpeakerVerificationScoring` calls
        it once per validation epoch, gathers the returned tensors across ranks
        and reduces them into EER and minDCF. A recipe that wants a different
        metric can reduce the same buffers from its own callback.

        Returns:
            ``(scores, labels)`` as 1-D tensors. Both are empty when no trial
            batch has been seen since the last call.
        """
        if not self.trial_scores:
            empty = torch.zeros(0)
            return empty, empty.to(torch.long)

        scores = torch.cat(self.trial_scores).flatten().float()
        labels = torch.cat(self.trial_labels).flatten().long()
        self.reset_trials()
        return scores, labels

    def reset_trials(self) -> None:
        """Drop every buffered trial score and label.

        Public for the same reason as :meth:`pop_trials`:
        :class:`espnet3.systems.spk.callbacks.SpeakerVerificationScoring` calls
        it at the start of every validation epoch, so that trials left behind
        by an interrupted epoch cannot leak into the next metric.
        """
        self.trial_scores.clear()
        self.trial_labels.clear()
