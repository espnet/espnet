import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.abs_sampler import AbsSampler

DEFAULT_EXCLUDED_KEY_PREFIXES = ("utt2category", "utt2fs")


class ChunkIterFactory(AbsIterFactory):
    """Creates chunks from a sequence

    Examples:
        >>> batches = [["id1"], ["id2"], ...]
        >>> batch_size = 128
        >>> chunk_length = 1000
        >>> iter_factory = ChunkIterFactory(dataset, batches, batch_size, chunk_length)
        >>> it = iter_factory.build_iter(epoch)
        >>> for ids, batch in it:
        ...     ...

    - The number of mini-batches are varied in each epochs and
      we can't get the number in advance
      because IterFactory doesn't be given to the length information.
    - Since the first reason, "num_iters_per_epoch" can't be implemented
      for this iterator. Instead of it, "num_samples_per_epoch" is implemented.

    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        chunk_length: Union[int, str],
        chunk_shift_ratio: float = 0.5,
        num_cache_chunks: int = 1024,
        num_samples_per_epoch: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        excluded_key_prefixes: Optional[List[str]] = None,
        default_fs: Optional[int] = None,
    ):
        assert check_argument_types()
        assert all(len(x) == 1 for x in batches), "batch-size must be 1"

        self.per_sample_iter_factory = SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            num_iters_per_epoch=num_samples_per_epoch,
            seed=seed,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        self.num_cache_chunks = max(num_cache_chunks, batch_size)
        if isinstance(chunk_length, str):
            if len(chunk_length) == 0:
                raise ValueError("e.g. 5,8 or 3-5: but got empty string")

            self.chunk_lengths = []
            for x in chunk_length.split(","):
                try:
                    sps = list(map(int, x.split("-")))
                except ValueError:
                    raise ValueError(f"e.g. 5,8 or 3-5: but got {chunk_length}")

                if len(sps) > 2:
                    raise ValueError(f"e.g. 5,8 or 3-5: but got {chunk_length}")
                elif len(sps) == 2:
                    # Append all numbers between the range into the candidates
                    self.chunk_lengths += list(range(sps[0], sps[1] + 1))
                else:
                    self.chunk_lengths += [sps[0]]
        else:
            # Single candidates: Fixed chunk length
            self.chunk_lengths = [chunk_length]

        self.chunk_shift_ratio = chunk_shift_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        # Default sampling frequency used to decide the chunk length
        # in case that different batches have different sampling frequencies
        # (If None, the chunk length is always fixed)
        self.default_fs = default_fs

        # keys that satisfy either condition below will be excluded from the length
        # consistency check:
        #  - exactly match one of the prefixes in `excluded_key_prefixes`
        #  - have one of the prefixes in `excluded_key_prefixes` and end with numbers
        if excluded_key_prefixes is None:
            excluded_key_prefixes = DEFAULT_EXCLUDED_KEY_PREFIXES
        else:
            for k in DEFAULT_EXCLUDED_KEY_PREFIXES:
                if k not in excluded_key_prefixes:
                    excluded_key_prefixes.append(k)
        self.excluded_key_pattern = (
            "(" + "[0-9]*)|(".join(excluded_key_prefixes) + "[0-9]*)"
        )
        if self.excluded_key_pattern:
            logging.info(
                f"Data keys with the following patterns will be excluded from the "
                f"length consistency check:\n{self.excluded_key_pattern}"
            )

    def build_iter(
        self,
        epoch: int,
        shuffle: Optional[bool] = None,
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor]]]:
        per_sample_loader = self.per_sample_iter_factory.build_iter(epoch, shuffle)

        if shuffle is None:
            shuffle = self.shuffle
        state = np.random.RandomState(epoch + self.seed)

        # NOTE(kamo):
        #   This iterator supports multiple chunk lengths and
        #   keep chunks for each lengths here until collecting specified numbers
        cache_chunks_dict = defaultdict(dict)
        cache_id_list_dict = defaultdict(dict)
        for ids, batch in per_sample_loader:
            # Must be per-sample-loader
            assert len(ids) == 1, f"Must be per-sample-loader: {len(ids)}"
            assert all(len(x) == 1 for x in batch.values())

            # Get keys of sequence data
            sequence_keys = []
            for key in batch:
                if key + "_lengths" in batch:
                    sequence_keys.append(key)
            # Remove lengths data and get the first sample
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            id_ = ids[0]

            for key in sequence_keys:
                if self.excluded_key_pattern is not None and re.fullmatch(
                    self.excluded_key_pattern, key
                ):
                    # ignore length inconsistency for `excluded_key_prefixes`
                    continue
                if len(batch[key]) != len(batch[sequence_keys[0]]):
                    raise RuntimeError(
                        f"All sequences must has same length: "
                        f"{len(batch[key])} != {len(batch[sequence_keys[0]])}"
                    )

            # Get sampling frequency of the batch to recalculate the chunk length
            fs = batch.get("utt2fs", torch.LongTensor([16000])).type(torch.int64).item()
            default_fs = fs if self.default_fs is None else self.default_fs
            assert fs % default_fs == 0 or default_fs % fs == 0

            L = len(batch[sequence_keys[0]])
            # Select chunk length
            chunk_lengths = [lg * fs // default_fs for lg in self.chunk_lengths]
            chunk_lengths = [lg for lg in chunk_lengths if lg < L]
            if len(chunk_lengths) == 0:
                logging.warning(
                    f"The length of '{id_}' is {L}, but it is shorter than "
                    f"any candidates of chunk-length: {self.chunk_lengths}"
                )
                continue

            # Convert numpy array to number
            category = (
                batch.get("utt2category", torch.LongTensor([0]))
                .type(torch.int64)
                .item()
            )

            W = int(state.choice(chunk_lengths, 1))
            cache_id_list = cache_id_list_dict[category].setdefault(W, [])
            cache_chunks = cache_chunks_dict[category].setdefault(W, {})

            # Shift width to the next chunk
            S = int(W * self.chunk_shift_ratio)
            # Number of chunks
            N = (L - W) // S + 1
            if shuffle:
                Z = state.randint(0, (L - W) % S + 1)
            else:
                Z = 0

            # Split a sequence into chunks.
            # Note that the marginal frames divided by chunk length are discarded
            for k, v in batch.items():
                if k not in cache_chunks:
                    cache_chunks[k] = []
                if k in sequence_keys:
                    # Shift chunks with overlapped length for data augmentation
                    if self.excluded_key_pattern is not None and re.fullmatch(
                        self.excluded_key_pattern, k
                    ):
                        for _ in range(N):
                            cache_chunks[k].append(v)
                    else:
                        cache_chunks[k] += [
                            v[Z + i * S : Z + i * S + W] for i in range(N)
                        ]
                else:
                    # If not sequence, use whole data instead of chunk
                    cache_chunks[k] += [v for _ in range(N)]
            cache_id_list += [id_ for _ in range(N)]

            if len(cache_id_list) > self.num_cache_chunks:
                cache_id_list, cache_chunks = yield from self._generate_mini_batches(
                    cache_id_list,
                    cache_chunks,
                    shuffle,
                    state,
                )

            cache_id_list_dict[category][W] = cache_id_list
            cache_chunks_dict[category][W] = cache_chunks

        else:
            for category in cache_id_list_dict.keys():
                for W in cache_id_list_dict[category]:
                    cache_id_list = cache_id_list_dict[category].setdefault(W, [])
                    cache_chunks = cache_chunks_dict[category].setdefault(W, {})

                    yield from self._generate_mini_batches(
                        cache_id_list,
                        cache_chunks,
                        shuffle,
                        state,
                    )

    def _generate_mini_batches(
        self,
        id_list: List[str],
        batches: Dict[str, List[torch.Tensor]],
        shuffle: bool,
        state: np.random.RandomState,
    ):
        if shuffle:
            indices = np.arange(0, len(id_list))
            state.shuffle(indices)
            batches = {k: [v[i] for i in indices] for k, v in batches.items()}
            id_list = [id_list[i] for i in indices]

        bs = self.batch_size
        while len(id_list) >= bs:
            # Make mini-batch and yield
            yield (
                id_list[:bs],
                {k: torch.stack(v[:bs], 0) for k, v in batches.items()},
            )
            id_list = id_list[bs:]
            batches = {k: v[bs:] for k, v in batches.items()}

        return id_list, batches
