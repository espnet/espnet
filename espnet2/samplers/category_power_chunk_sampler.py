"""Mini-batches of chunks drawn per epoch with power-law category weights."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from typeguard import typechecked

from espnet2.samplers.abs_sampler import AbsSampler

Chunk = Tuple[str, int, int]


class CategoryPowerChunkSampler(AbsSampler):
    r"""Draws the chunks of each epoch so that categories are balanced.

    Used by ``--chunk_random_access`` with ``--chunk_upsampling_factor``. Each
    chunk ``(sequence id, start, end)`` belongs to the category of its sequence
    (``utt2category``). With :math:`n_c` chunks in category :math:`c`,
    :math:`N = \sum_c n_c` and :math:`\beta` the upsampling factor, an epoch has
    :math:`N` chunks, of which category :math:`c` gets

    .. math:: m_c \approx N \frac{n_c^\beta}{\sum_{c'} n_{c'}^\beta}.

    :math:`\beta = 1` keeps every chunk once (the sizes of the categories),
    :math:`\beta = 0` gives every category the same number of chunks. A category
    with :math:`m_c > n_c` repeats all its chunks :math:`\lfloor m_c / n_c
    \rfloor` times and draws the rest without replacement; one with
    :math:`m_c < n_c` draws :math:`m_c` of its chunks without replacement, a
    different subset each epoch.

    The drawn chunks are shuffled and split into mini-batches of about
    ``batch_size`` chunks, as UnsortedBatchSampler splits its keys. ``generate``
    draws for one seed (the iterator passes the epoch).

    Args:
        chunks: All chunks, (sequence id, start, end).
        utt2category: Category of each sequence id.
        batch_size: Chunks per mini-batch.
        upsampling_factor: beta, in [0, 1].
        num_batches: Keep only the first mini-batches of each epoch.
        seed: Seed used by ``__iter__``.
        rank, world_size: With distributed training, each rank keeps the
            chunks ``rank::world_size`` of every mini-batch (all ranks draw
            the same mini-batches, as they share the seed).
    """

    @typechecked
    def __init__(
        self,
        chunks: Sequence[Chunk],
        utt2category: Dict[str, str],
        batch_size: int,
        upsampling_factor: float,
        num_batches: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0.0 <= upsampling_factor <= 1.0:
            raise ValueError(
                f"upsampling_factor must be in [0, 1]: {upsampling_factor}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {batch_size}")
        if len(chunks) == 0:
            raise ValueError("no chunks")
        self.category2chunks: Dict[str, List[Chunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk[0] not in utt2category:
                raise KeyError(f"{chunk[0]} is not in utt2category")
            self.category2chunks[utt2category[chunk[0]]].append(chunk)
        self.categories = sorted(self.category2chunks)
        sizes = np.array([len(self.category2chunks[c]) for c in self.categories])
        weights = sizes.astype(np.float64) ** upsampling_factor
        self.num_drawn = self._round_to_total(
            len(chunks) * weights / weights.sum(), len(chunks)
        )
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        if batch_size < world_size:
            raise RuntimeError(
                f"The batch-size must be equal or more than world_size: "
                f"{batch_size} < {world_size}"
            )

    @staticmethod
    def _round_to_total(x: np.ndarray, total: int) -> Dict[int, int]:
        """Largest-remainder rounding of x to integers that sum to total."""
        floor = np.floor(x).astype(int)
        order = np.argsort(-(x - floor), kind="stable")
        floor[order[: total - floor.sum()]] += 1
        return {i: int(n) for i, n in enumerate(floor)}

    def category_counts(self) -> Dict[str, Tuple[int, int]]:
        """(chunks in the data, chunks drawn per epoch) of each category."""
        return {
            c: (len(self.category2chunks[c]), self.num_drawn[i])
            for i, c in enumerate(self.categories)
        }

    def generate(self, seed: int) -> List[List[Chunk]]:
        rng = np.random.RandomState(seed)
        drawn: List[Chunk] = []
        for i, c in enumerate(self.categories):
            pool = self.category2chunks[c]
            repeats, rest = divmod(self.num_drawn[i], len(pool))
            idx = [rng.permutation(len(pool)) for _ in range(repeats)]
            idx.append(rng.choice(len(pool), rest, replace=False))
            drawn.extend(pool[j] for j in np.concatenate(idx))
        order = rng.permutation(len(drawn))
        drawn = [drawn[j] for j in order]
        # the split of UnsortedBatchSampler (drop_last=False)
        n = max(len(drawn) // self.batch_size, 1)
        batches = [
            drawn[k * len(drawn) // n : (k + 1) * len(drawn) // n] for k in range(n)
        ]
        if self.num_batches is not None:
            batches = batches[: self.num_batches]
        if self.world_size > 1:
            batches = [b[self.rank :: self.world_size] for b in batches]
        return batches

    def __len__(self) -> int:
        n = max(sum(self.num_drawn.values()) // self.batch_size, 1)
        return n if self.num_batches is None else min(n, self.num_batches)

    def __iter__(self):
        return iter(self.generate(self.seed))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N-categories={len(self.categories)}, "
            f"batch_size={self.batch_size}, num_batches={len(self)})"
        )
