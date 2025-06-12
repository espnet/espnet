import collections
import copy
import functools
import json
import logging
import numbers
import random
import re
import types  # noqa
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import h5py
import humanfriendly
import kaldiio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typeguard import typechecked

from espnet2.fileio.multi_sound_scp import MultiSoundScpReader
from espnet2.fileio.npy_scp import NpyScpReader
from espnet2.fileio.rand_gen_dataset import (
    FloatRandomGenerateDataset,
    IntRandomGenerateDataset,
)
from espnet2.fileio.read_text import (
    RandomTextReader,
    load_num_sequence_text,
    read_2columns_text,
    read_label,
)
from espnet2.fileio.rttm import RttmReader
from espnet2.fileio.score_scp import SingingScoreReader
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.utils.sized_dict import SizedDict


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(
        self,
        loader,
        dtype: Union[None, str] = None,
        allow_multi_rates: bool = False,
    ):
        self.loader = loader
        self.dtype = dtype
        self.rate = None
        self.allow_multi_rates = allow_multi_rates

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )

            if not self.allow_multi_rates and (
                self.rate is not None and self.rate != rate
            ):
                raise RuntimeError(
                    f"Sampling rates are mismatched: {self.rate} != {rate}"
                )
            self.rate = rate
            # Multichannel wave fie
            # array: (NSample, Channel) or (Nsample)
            if self.dtype is not None:
                array = array.astype(self.dtype)

        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval
            if self.dtype is not None:
                array = array.astype(self.dtype)

        assert isinstance(array, np.ndarray), type(array)
        return array


class H5FileWrapper:
    @typechecked
    def __init__(self, path: str):
        self.path = path
        self.h5_file = h5py.File(path, "r")

    def __repr__(self) -> str:
        return str(self.h5_file)

    def __len__(self) -> int:
        return len(self.h5_file)

    def __iter__(self):
        return iter(self.h5_file)

    def __getitem__(self, key) -> np.ndarray:
        value = self.h5_file[key]
        return value[()]


class AdapterForSingingScoreScpReader(collections.abc.Mapping):
    def __init__(self, loader):
        self.loader = loader

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]
        assert (
            len(retval) == 3
            and isinstance(retval["tempo"], int)
            and isinstance(retval["note"], list)
        )
        tempo = retval["tempo"]

        return tempo, retval["note"]


class AdapterForLabelScpReader(collections.abc.Mapping):
    @typechecked
    def __init__(self, loader: Dict[str, List[List[Union[str, float, int]]]]):
        self.loader = loader

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        assert isinstance(retval, list)
        seq_len = len(retval)
        sample_time = np.zeros((seq_len, 2))
        sample_label = []
        for i in range(seq_len):
            sample_time[i, 0] = np.float32(retval[i][0])
            sample_time[i, 1] = np.float32(retval[i][1])
            sample_label.append(retval[i][2])

        assert isinstance(sample_time, np.ndarray) and isinstance(sample_label, list)
        return sample_time, sample_label


def sound_loader(path, float_dtype=None, multi_columns=False, allow_multi_rates=False):
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    loader = SoundScpReader(
        path, always_2d=False, dtype=float_dtype, multi_columns=multi_columns
    )

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader, allow_multi_rates=allow_multi_rates)


def multi_columns_sound_loader(path, float_dtype=None, allow_multi_rates=False):
    return sound_loader(
        path, float_dtype, multi_columns=True, allow_multi_rates=allow_multi_rates
    )


def variable_columns_sound_loader(path, float_dtype=None, allow_multi_rates=False):
    # The file is as follows:
    #   utterance_id_A /some/where/a1.wav /some/where/a2.wav /some/where/a3.wav
    #   utterance_id_B /some/where/b1.flac /some/where/b2.flac

    # NOTE(wangyou): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(wangyou): The audio signal is normalized to [-1,1] range.
    loader = MultiSoundScpReader(path, always_2d=False, dtype=float_dtype, stack_axis=0)
    return AdapterForSoundScpReader(loader, allow_multi_rates=allow_multi_rates)


def score_loader(path):
    loader = SingingScoreReader(fname=path)
    return AdapterForSingingScoreScpReader(loader)


def label_loader(path):
    loader = read_label(path)
    return AdapterForLabelScpReader(loader)


def kaldi_loader(
    path, float_dtype=None, max_cache_fd: int = 0, allow_multi_rates=False
):
    loader = kaldiio.load_scp(path, max_cache_fd=max_cache_fd)
    return AdapterForSoundScpReader(
        loader, float_dtype, allow_multi_rates=allow_multi_rates
    )


def rand_int_loader(filepath, loader_type):
    # e.g. rand_int_3_10
    try:
        low, high = map(int, loader_type[len("rand_int_") :].split("_"))
    except ValueError:
        raise RuntimeError(f"e.g rand_int_3_10: but got {loader_type}")
    return IntRandomGenerateDataset(filepath, low, high)


DATA_TYPES = {
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    ),
    "multi_columns_sound": dict(
        func=multi_columns_sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
        help="Enable multi columns wav.scp. "
        "The following text file can be loaded as multi channels audio data"
        "\n\n"
        "   utterance_id_a a.wav a2.wav\n"
        "   utterance_id_b b.wav b2.wav\n"
        "   ...",
    ),
    "variable_columns_sound": dict(
        func=variable_columns_sound_loader,
        kwargs=["float_dtype", "allow_multi_rates"],
        help="Loading variable numbers (columns) of audios in wav.scp. "
        "The following text file can be loaded as stacked audio data"
        "\n\n"
        "   utterance_id_a a1.wav a2.wav a3.wav\n"
        "   utterance_id_b b1.wav\n"
        "   utterance_id_c c1.wav c2.wav\n"
        "   ...\n\n"
        "Note that audios of different lengths will be right-padded with np.nan "
        "to the longest audio in the sample.\n"
        "A preprocessor must be used to remove these paddings.",
    ),
    "score": dict(
        func=score_loader,
        kwargs=[],
        help="Return text as is. The text contains tempo and note info.\n"
        "For each note, 'start' 'end' 'syllabel' 'midi' and 'phones' are included. "
        "\n\n"
        "   utterance_id_A tempo_a start_1 end_1 syllable_1 midi_1 phones_1 ...\n"
        "   utterance_id_B tempo_b start_1 end_1 syllable_1 midi_1 phones_1 ...\n"
        "   ...",
    ),
    "duration": dict(
        func=label_loader,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A start_1 end_1 phone_1 start_2 end_2 phone_2 ...\n"
        "   utterance_id_B start_1 end_1 phone_1 start_2 end_2 phone_2 ...\n"
        "   ...",
    ),
    "kaldi_ark": dict(
        func=kaldi_loader,
        kwargs=["max_cache_fd", "allow_multi_rates"],
        help="Kaldi-ark file type."
        "\n\n"
        "   utterance_id_A /some/where/a.ark:123\n"
        "   utterance_id_B /some/where/a.ark:456\n"
        "   ...",
    ),
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ...",
    ),
    "text_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12 0 1 3\n"
        "   utterance_id_B 3 3 1\n"
        "   ...",
    ),
    "csv_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 100,80\n"
        "   utterance_id_B 143,80\n"
        "   ...",
    ),
    "text_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12. 3.1 3.4 4.4\n"
        "   utterance_id_B 3. 3.12 1.1\n"
        "   ...",
    ),
    "csv_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 12.,3.1,3.4,4.4\n"
        "   utterance_id_B 3.,3.12,1.1\n"
        "   ...",
    ),
    "text": dict(
        func=read_2columns_text,
        kwargs=["keys_to_load"],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A hello world\n"
        "   utterance_id_B foo bar\n"
        "   ...",
    ),
    "random_text": dict(
        func=RandomTextReader,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   hello world\n"
        "   foo bar\n"
        "   ...",
    ),
    "hdf5": dict(
        func=H5FileWrapper,
        kwargs=[],
        help="A HDF5 file which contains arrays at the first level or the second level."
        "   >>> f = h5py.File('file.h5')\n"
        "   >>> array1 = f['utterance_id_A']\n"
        "   >>> array2 = f['utterance_id_B']\n",
    ),
    "rand_float": dict(
        func=FloatRandomGenerateDataset,
        kwargs=[],
        help="Generate random float-ndarray which has the given shapes "
        "in the file."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rand_int_\\d+_\\d+": dict(
        func=rand_int_loader,
        kwargs=["loader_type"],
        help="e.g. 'rand_int_0_10'. Generate random int-ndarray which has the given "
        "shapes in the path. "
        "Give the lower and upper value by the file type. e.g. "
        "rand_int_0_10 -> Generate integers from 0 to 10."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rttm": dict(
        func=RttmReader,
        kwargs=[],
        help="rttm file loader, currently support for speaker diarization"
        "\n\n"
        "    SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>"
        "    SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>"
        "    SPEAKER file1 3 500 4023 <NA> <NA> spk1 <NA>"
        "    END     file1 <NA> 4023 <NA> <NA> <NA> <NA>"
        "   ...",
    ),
}


class AbsDataset(Dataset, ABC):
    @abstractmethod
    def has_name(self, name) -> bool:
        raise NotImplementedError

    @abstractmethod
    def names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, uid) -> Tuple[Any, Dict[str, np.ndarray]]:
        raise NotImplementedError


class ESPnetDataset(AbsDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
        {'input': per_utt_array, 'output': per_utt_array}
    """

    @typechecked
    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Optional[
            Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        allow_multi_rates: bool = False,
        keys_to_load: Optional[Set[Union[str, int]]] = None,
    ):
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.max_cache_fd = max_cache_fd
        # allow audios to have different sampling rates
        self.allow_multi_rates = allow_multi_rates

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            if name in self.loader_dict:
                raise RuntimeError(f'"{name}" is duplicated for data-key')

            loader = self._build_loader(path, _type, keys_to_load)
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

    def _build_loader(
        self,
        path: str,
        loader_type: str,
        keys_to_load: Optional[Set[Union[str, int]]],
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
            keys_to_load:  The set of keys to load. If None, load all.
        """
        for key, dic in DATA_TYPES.items():
            # e.g. loader_type="sound"
            # -> return DATA_TYPES["sound"]["func"](path)
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    elif key2 == "allow_multi_rates":
                        kwargs["allow_multi_rates"] = self.allow_multi_rates
                    elif key2 == "keys_to_load":
                        kwargs["keys_to_load"] = keys_to_load
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                try:
                    return func(path, **kwargs)
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    def has_name(self, name) -> bool:
        return name in self.loader_dict

    def names(self) -> Tuple[str, ...]:
        return tuple(self.loader_dict)

    def __iter__(self):
        return iter(next(iter(self.loader_dict.values())))

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    @typechecked
    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:

        # Change integer-id to string-id
        if isinstance(uid, int):
            d = next(iter(self.loader_dict.values()))
            uid = list(d)[uid]

        if self.cache is not None and uid in self.cache:
            data = self.cache[uid]
            return uid, data

        data = {}
        # 1. Load data from each loaders
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                if isinstance(value, (list)):
                    value = np.array(value)
                if not isinstance(
                    value, (np.ndarray, torch.Tensor, str, numbers.Number, tuple)
                ):
                    raise TypeError(
                        (
                            "Must be ndarray, torch.Tensor, "
                            "str,  Number or tuple: {}".format(type(value))
                        )
                    )
            except Exception:
                path, _type = self.debug_info[name]
                logging.error(
                    f"Error happened with path={path}, type={_type}, id={uid}"
                )
                raise

            # torch.Tensor is converted to ndarray
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            elif isinstance(value, numbers.Number):
                value = np.array([value])
            data[name] = value

        # 2. [Option] Apply preprocessing
        if getattr(self, "install_speaker_prompt", None) is not None:
            self.install_speaker_prompt(uid, data)
        #   e.g. espnet2.train.preprocessor:CommonPreprocessor
        if self.preprocess is not None:
            key_prefix = self.task + " " if hasattr(self, "task") else ""
            data = self.preprocess(key_prefix + uid, data)

        # 3. Force data-precision
        for name in data:
            value = data[name]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    f"All values must be converted to np.ndarray object "
                    f'by preprocessing, but "{name}" is still {type(value)}.'
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
        return retval


class ESPnetSpeechLMDataset(ESPnetDataset):
    """ESPnet Speech LM Dataset.

    Dataset object that is specifically designed for SpeechLM. It will allows
    dataset-level operations (e.g., on-the-fly speaker prompt sampling). It is
    task-specific and can be queried by ESPnetMultiTaskDataset.
    """

    def __init__(
        self,
        example_list: List,
        task: str,
        **kwargs,
    ):
        super(ESPnetSpeechLMDataset, self).__init__(**kwargs)

        # (1) build spk2utt map
        if "utt2spk" in self.loader_dict:
            self.spk2utt = {}
            for k, v in self.loader_dict["utt2spk"].items():
                if v not in self.spk2utt:
                    self.spk2utt[v] = []
                self.spk2utt[v].append(k)

        # (2) keep example_list and clean some non-iterable loaders
        example_dict = {k: None for k in example_list}  # hash for faster query
        for key in self.loader_dict.keys():
            loader = self.loader_dict[key]
            if isinstance(loader, Dict):
                loader = {k: v for k, v in loader.items() if k in example_dict}
                self.loader_dict[key] = loader

        # (3) keep task
        self.task = task

    def install_speaker_prompt(self, uid: str, data: Dict):
        """Assume the names are utt2spk and wav.scp. Hard code here."""
        if "utt2spk" in self.loader_dict:
            spk = self.loader_dict["utt2spk"][uid]
            utts = self.spk2utt[spk]

            if len(utts) == 1:  # at least itself
                utt = utts[0]
            else:
                while True:
                    utt = random.sample(utts, 1)[0]
                    if uid != utt:
                        break

            if "wav.scp" not in self.loader_dict:
                raise ValueError("speaker prompt is sampled from wav.scp loader")

            data["utt2spk"] = self.loader_dict["wav.scp"][utt]


class ESPnetMultiTaskDataset(AbsDataset):
    """ESPnet Multi Task Dataset.

    The top-level Dataset object that can manage multiple ESPnetSpeechLMDataset
    objects, each of which serves a specific task and dataset.
    This object will query all these ESPnetSpeechLMDataset and combine examples
    from different tasks for multi-task training. Typically, this dataset is
    used in ESPnet SpeechLM models
    See details in:
    <espnet>/egs2/TEMPLATE/speechlm1#data-loading-and-preprocessing
    """

    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        key_file: str = None,
        **kwargs,
    ):
        if key_file is not None:
            self.key_dict = {line.strip().split()[0]: None for line in open(key_file)}
        else:
            self.key_dict = None

        self.iterator_map = {}
        self.datasets = []
        for triplet in path_name_type_list:
            path, _, _type = triplet
            assert _type == "dataset_json", f"Non-Json triplet: {triplet}"
            json_dict = json.load(open(path))

            this_path_name_type_list = []
            for triplet in json_dict["data_files"]:
                path, _, _type = triplet.strip().split(",")
                # use the stem file name as the name
                this_path_name_type_list.append(
                    (
                        path,
                        path.split("/")[-1],
                        _type,
                    )
                )

            # example_list is for sub_dataest -> no task prefix
            example_list = [line.strip().split()[0] for line in open(path)]
            if self.key_dict is not None:
                example_list = [
                    e
                    for e in example_list
                    if json_dict["task"] + "_" + e in self.key_dict
                ]

            dataset = ESPnetSpeechLMDataset(
                path_name_type_list=this_path_name_type_list,
                example_list=example_list,
                task=json_dict["task"],
                **kwargs,
            )
            self.datasets.append(dataset)

            # iterator_map is for merged dataset -> with task prefix
            self.iterator_map.update(
                {json_dict["task"] + "_" + e: dataset for e in example_list}
            )

        self.encoder_decoder_format = getattr(
            kwargs["preprocess"], "encoder_decoder_format", False
        )
        self.apply_utt2category = False
        self.example_list = list(self.iterator_map.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
        iterator = self.iterator_map[uid]
        uid_without_prefix = uid.lstrip(iterator.task + "_")
        uid, data = iterator[uid_without_prefix]
        uid = iterator.task + "_" + uid
        return uid, data

    # Keep same interface with IterableDataset
    def has_name(self, name) -> bool:
        return name in self.names()

    def names(self) -> Tuple[str, ...]:
        if self.encoder_decoder_format:
            return ("enc_seq", "dec_seq", "prefix_len")
        else:
            return ("dec_seq", "prefix_len")

    def __repr__(self):
        string = "##### Multi-Task Dataset #####\n"
        for idx, dataset in enumerate(self.datasets):
            string += f"## Sub-Dataset: {idx}; Task: {dataset.task} ##\n"
            string += f"{dataset}\n"
        return string

    def __len__(self):
        return sum([len(d.example_list) for d in self.datasets])
