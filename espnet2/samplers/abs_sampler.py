from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Iterator
from typing import Sequence
from typing import Tuple
from typing import Union

import h5py
from torch.utils.data import Sampler

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.utils.hdf5_corpus import H5FileWrapper


class AbsSampler(Sampler, ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        raise NotImplementedError

    @staticmethod
    def _load_shape_files(
        shape_files: Sequence[Union[Path, str, h5py.Group]], sort: bool,
    ):
        if isinstance(shape_files[0], h5py.Group):
            utt2shapes = [H5FileWrapper(s) for s in shape_files]
        else:
            # utt2shape: (Length, ...)
            #    uttA 100,...
            #    uttB 201,...
            utt2shapes = [
                load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
            ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if len(d) != len(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )
            for k in first_utt2shape:
                if k not in d:
                    raise RuntimeError(
                        f"keys are mismatched between {s} != {shape_files[0]}"
                    )

        if sort:
            # Sort samples in ascending order
            # (shape order should be like (Length, Dim))
            keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
            if len(keys) == 0:
                raise RuntimeError(f"0 lines found: {shape_files[0]}")
        else:
            keys = list(utt2shapes[0])
        return utt2shapes, keys
