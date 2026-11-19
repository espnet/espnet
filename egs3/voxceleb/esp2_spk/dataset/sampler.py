"""Speaker-balanced batches, as ESPnet2's ``iterator_type: category``."""

from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import BatchSampler, Sampler


class SpeakerBalancedBatchSampler(BatchSampler):
    """Group a sampler's indices into batches with no speaker repeated.

    This is ESPnet2's ``CategoryBalancedSampler`` (``batch_type: catbel``),
    which its VoxCeleb recipes train with, written as a ``BatchSampler`` over
    an index sampler so that Lightning can shard it: under DDP, Lightning
    rebuilds a ``BatchSampler`` subclass with its ``sampler`` replaced by a
    ``DistributedSampler`` and keeps the other arguments, and calls
    ``set_epoch`` on that sampler every epoch. Each rank then batches its own
    shard, so a speaker is unique within a rank's batch. That is all the
    AAM-softmax losses need, since they score every utterance against the
    class weights and never against the rest of the batch.

    ESPnet2 drops an utterance whose speaker is already in the batch being
    built (~2% per epoch on VoxCeleb 1+2 with speed perturbation and a batch of
    512). Here it is held over for a later batch instead, so every utterance
    is used and an epoch is exactly ``len(sampler) // batch_size`` batches,
    which Lightning relies on: it reads the length once and validates on the
    batch that reaches it. Only the final batch of an epoch may repeat a
    speaker, when the held-over utterances cannot be placed before it.

    Lightning only records the arguments of classes already imported when it
    builds the dataloader, so the recipe's ``dataset`` package imports this
    module.

    Args:
        sampler: Yields dataset indices, e.g. a ``RandomSampler``.
        batch_size: Utterances per batch, per rank.
        drop_last: Whether to drop the final partial batch.
        dataset: A ``CombinedDataset`` whose datasets provide
            ``speaker_labels()``, or a single such dataset.

    Raises:
        ValueError: If the dataset provides no speaker labels.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool,
        dataset,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = dataset
        parts = getattr(dataset, "datasets", [dataset])
        labels: Sequence[str] = [
            label for part in parts for label in part.speaker_labels()
        ]
        if not labels:
            raise ValueError("SpeakerBalancedBatchSampler needs speaker labels")
        _, codes = np.unique(np.asarray(labels), return_inverse=True)
        self._codes = codes.astype(np.int64).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        codes = self._codes
        batch: List[int] = []
        speakers: set = set()
        held: List[int] = []

        def fill_from_held() -> None:
            nonlocal held
            kept = []
            for index in held:
                if len(batch) < self.batch_size and codes[index] not in speakers:
                    batch.append(index)
                    speakers.add(codes[index])
                else:
                    kept.append(index)
            held = kept

        for index in self.sampler:
            if codes[index] in speakers:
                held.append(index)
            else:
                batch.append(index)
                speakers.add(codes[index])
            while len(batch) == self.batch_size:
                yield batch
                batch, speakers = [], set()
                fill_from_held()

        # What is left cannot be balanced further; it fills the last batches.
        rest = batch + held
        for start in range(0, len(rest), self.batch_size):
            chunk = rest[start : start + self.batch_size]
            if len(chunk) == self.batch_size or not self.drop_last:
                yield chunk
