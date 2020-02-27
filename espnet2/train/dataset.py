import collections
import copy
import logging
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

import h5py
import humanfriendly
import kaldiio
import numpy as np
from torch.utils.data.dataset import Dataset
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.utils.fileio import load_num_sequence_text
from espnet2.utils.fileio import NpyScpReader
from espnet2.utils.fileio import read_2column_text
from espnet2.utils.fileio import SoundScpReader
from espnet2.utils.rand_gen_dataset import FloatRandomGenerateDataset
from espnet2.utils.rand_gen_dataset import IntRandomGenerateDataset
from espnet2.utils.sized_dict import SizedDict


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(self, loader: SoundScpReader, dtype=None):
        assert check_argument_types()
        self.loader = loader
        self.dtype = dtype
        self.rate = None

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        rate, array = self.loader[key]
        if self.rate is not None and self.rate != rate:
            raise RuntimeError(f"Sampling rates are mismatched: {self.rate} != {rate}")
        self.rate = rate
        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        return array


class ESPnetDataset(Dataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
        {'input': per_utt_array, 'output': per_utt_array}
    """

    def __init__(
        self,
        path_name_type_list: Sequence[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
    ):
        assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            if name in self.loader_dict:
                raise RuntimeError(f'"{name}" is duplicated for data-key')

            loader = self._create_loader(path, _type)
            self.loader_dict[name] = loader
            self.debug_info[name] = path, _type
            if len(self.loader_dict[name]) == 0:
                raise RuntimeError(f"{path} has no samples")

            # TODO(kamo): Should check consistency of each utt-keys?

        if isinstance(max_cache_size, str):
            max_cache_size = humanfriendly.parse_size(max_cache_size)
        self.max_cache_size = max_cache_size
        if max_cache_size > 0:
            self.cache = SizedDict(shared=True)
        else:
            self.cache = None

    def _create_loader(
        self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, str]]:
        """Helper function to instantiate Loader

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """

        if loader_type == "sound":
            # path looks like:
            #   utta /some/where/a.wav
            #   uttb /some/where/a.flac

            # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
            # like Kaldi e.g. "cat a.wav |".
            # NOTE(kamo): The audio signal is normalized to [-1,1] range.
            loader = SoundScpReader(path, normalize=True, always_2d=False)

            # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
            # but ndarray is desired, so Adapter class is inserted here
            return AdapterForSoundScpReader(loader, self.float_dtype)

        elif loader_type == "pipe_wav":
            # path looks like:
            #   utta cat a.wav |
            #   uttb cat b.wav |

            # NOTE(kamo): I don't think this case is practical
            # because subprocess takes much times due to fork().

            # NOTE(kamo): kaldiio doesn't normalize the signal.
            loader = kaldiio.load_scp(path)
            return AdapterForSoundScpReader(loader, self.float_dtype)

        elif loader_type == "kaldi_ark":
            # path looks like:
            #   utta /some/where/a.ark:123
            #   uttb /some/where/a.ark:456
            return kaldiio.load_scp(path)

        elif loader_type == "npy":
            # path looks like:
            #   utta /some/where/a.npy
            #   uttb /some/where/b.npy
            raise NpyScpReader(path)

        elif loader_type == "hdf5":
            raise h5py.File(path, "r")

        elif loader_type in ("text_int", "text_float", "csv_int", "csv_float"):
            # Not lazy loader, but as vanilla-dict
            return load_num_sequence_text(path, loader_type)

        elif loader_type == "text":
            # NOTE(kamo): Return str instead of np.ndarray in this case.
            #  Must be converted by "preprocess"
            return read_2column_text(path)

        elif loader_type == "rand_float":
            return FloatRandomGenerateDataset(path)

        elif loader_type.startswith("rand_int"):
            # e.g. rand_int_3_10
            try:
                low, high = loader_type[len("rand_int") + 1 :].split("_")
                low, high = int(low), int(high)
            except ValueError:
                raise RuntimeError(f"e.g rand_int_3_10: but got {loader_type}")
            return IntRandomGenerateDataset(path, low, high)

        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    def has_name(self, name) -> bool:
        return name in self.loader_dict

    def names(self) -> Tuple[str, ...]:
        return tuple(self.loader_dict)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += f"("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def __len__(self):
        raise RuntimeError(
            "This method doesn't be needed because " "we use custom BatchSampler "
        )

    # NOTE(kamo):
    # Typically pytorch's Dataset.__getitem__ accepts an inger index,
    # however this Dataset handle a string, which represents a sample-id.
    def __getitem__(self, uid: str) -> Tuple[str, Dict[str, np.ndarray]]:
        assert check_argument_types()

        if self.cache is not None and uid in self.cache:
            data = self.cache[uid]
            return uid, data

        data = {}
        # 1. Load data from each loaders
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                if not isinstance(value, (np.ndarray, str)):
                    raise TypeError(f"Must be ndarray or str: {type(value)}")
            except Exception:
                path, _type = self.debug_info[name]
                logging.error(
                    f"Error happened with path={path}, type={_type}, id={uid}"
                )
                raise
            data[name] = value

        # 2. [Option] Apply preprocessing
        #   e.g. espnet2.train.preprocessor:CommonPreprocessor
        if self.preprocess is not None:
            data = self.preprocess(uid, data)

        # 3. Force data-precision
        for name in self.loader_dict:
            value = data[name]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    f"str type object must be converted to np.ndarray object "
                    f'by preprocessing, but "{name}" is still str.'
                )

            # Cast to desired type
            if value.dtype.kind == "f":
                value = value.astype(self.float_dtype)
            elif value.dtype.kind == "i":
                value = value.astype(self.int_dtype)
            else:
                raise NotImplementedError(f"Not supported dtype: {value.dtype}")
            data[name] = value

        if self.cache is not None and self.cache.size < self.max_cache_size:
            self.cache[uid] = data

        retval = uid, data
        assert check_return_type(retval)
        return retval
