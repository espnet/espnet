"""Iterable dataset module."""

import copy
import json
from io import StringIO
from pathlib import Path
from typing import Callable, Collection, Dict, Iterator, List, Optional, Tuple, Union

import kaldiio
import numpy as np
import soundfile
import torch
from torch.utils.data.dataset import IterableDataset
from typeguard import typechecked

from espnet2.train.dataset import ESPnetDataset


def load_kaldi(input):
    retval = kaldiio.load_mat(input)
    if isinstance(retval, tuple):
        assert len(retval) == 2, len(retval)
        if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
            # sound scp case
            rate, array = retval
        elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
            # Extended ark format case
            array, rate = retval
        else:
            raise RuntimeError(f"Unexpected type: {type(retval[0])}, {type(retval[1])}")

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


DATA_TYPES = {
    "sound": lambda x: soundfile.read(x)[0],
    "multi_columns_sound": lambda x: np.concatenate(
        [soundfile.read(xx, always_2d=True)[0] for xx in x.split()], axis=1
    ),
    "variable_columns_sound": lambda x: np.stack(
        [soundfile.read(xx)[0] for xx in x.split()], axis=0
    ),
    "kaldi_ark": load_kaldi,
    "npy": np.load,
    "text_int": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.int64, delimiter=" "
    ),
    "csv_int": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.int64, delimiter=","
    ),
    "text_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=" "
    ),
    "csv_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=","
    ),
    "text": lambda x: x,
}


class IterableESPnetDataset(IterableDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = IterableESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                                  ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
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
        key_file: Optional[Union[str, List]] = None,
        preprocess_prefix: Optional[str] = None,
    ):
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.key_file = key_file
        self.preprocess_prefix = (
            preprocess_prefix if preprocess_prefix is not None else ""
        )

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        for path, name, _type in path_name_type_list:
            if name in self.debug_info:
                raise RuntimeError(f'"{name}" is duplicated for data-key')
            self.debug_info[name] = path, _type
            if _type not in DATA_TYPES:
                non_iterable_list.append((path, name, _type))
            else:
                self.path_name_type_list.append((path, name, _type))

        if len(non_iterable_list) != 0:
            # Some types doesn't support iterable mode
            self.non_iterable_dataset = ESPnetDataset(
                path_name_type_list=non_iterable_list,
                preprocess=preprocess,
                float_dtype=float_dtype,
                int_dtype=int_dtype,
            )
        else:
            self.non_iterable_dataset = None

        if Path(Path(path_name_type_list[0][0]).parent, "utt2category").exists():
            self.apply_utt2category = True
        else:
            self.apply_utt2category = False

    def has_name(self, name) -> bool:
        return name in self.debug_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.debug_info)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def __iter__(self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:
        if self.key_file is not None:
            if isinstance(self.key_file, str):
                uid_iter = (
                    line.rstrip().split(maxsplit=1)[0]
                    for line in open(self.key_file, encoding="utf-8")
                )
            else:
                uid_iter = self.key_file
        elif len(self.path_name_type_list) != 0:
            uid_iter = (
                line.rstrip().split(maxsplit=1)[0]
                for line in open(self.path_name_type_list[0][0], encoding="utf-8")
            )
        else:
            uid_iter = iter(self.non_iterable_dataset)

        files = [open(lis[0], encoding="utf-8") for lis in self.path_name_type_list]

        worker_info = torch.utils.data.get_worker_info()

        linenum = 0
        count = 0

        for count, uid in enumerate(uid_iter, 1):
            # If num_workers>=1, split keys
            if worker_info is not None:
                if (count - 1) % worker_info.num_workers != worker_info.id:
                    continue

            # 1. Read a line from each file
            while True:
                keys = []
                values = []
                for f in files:
                    linenum += 1
                    try:
                        line = next(f)
                    except StopIteration:
                        raise RuntimeError(f"{uid} is not found in the files")
                    sps = line.rstrip().split(maxsplit=1)
                    if len(sps) != 2:
                        raise RuntimeError(
                            f"This line doesn't include a space:"
                            f" {f}:L{linenum}: {line})"
                        )
                    key, value = sps
                    keys.append(key)
                    values.append(value)

                for k_idx, k in enumerate(keys):
                    if k != keys[0]:
                        raise RuntimeError(
                            f"Keys are mismatched. Text files (idx={k_idx}) is "
                            f"not sorted or not having same keys at L{linenum}"
                        )

                # If the key is matched, break the loop
                if len(keys) == 0 or keys[0] == uid:
                    break

            # 2. Load the entry from each line and create a dict
            data = {}
            # 2.a. Load data streamingly
            for value, (path, name, _type) in zip(values, self.path_name_type_list):
                func = DATA_TYPES[_type]
                # Load entry
                array = func(value)
                data[name] = array
            if self.non_iterable_dataset is not None:
                # 2.b. Load data from non-iterable dataset
                _, from_non_iterable = self.non_iterable_dataset[uid]
                data.update(from_non_iterable)

            # 3. [Option] Apply preprocessing
            #   e.g. espnet2.train.preprocessor:CommonPreprocessor
            if self.preprocess is not None:
                data = self.preprocess(self.preprocess_prefix + uid, data)

            # 4. Force data-precision
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

            yield uid, data

        if count == 0:
            raise RuntimeError("No iteration")


class SplicedIterableESPnetDataset(IterableDataset):
    """A data iterator that is spliced from multiple IterableESPnetDataset"""

    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        key_file: str = None,
        **kwargs,
    ):
        if key_file is not None:
            key_dict = {
                key.strip().split()[0]: None for key in open(key_file, encoding="utf-8")
            }
        else:
            key_dict = None

        self.data_iterators = []
        self.task_map = {}
        self.speaker_prompt_config = {}
        for triplet in path_name_type_list:
            path, _, _type = triplet
            assert _type == "json", f"Non-Json triplet: {triplet}"
            json_dict = json.load(open(path))

            iterator_path_name_type_list = []
            use_speaker_prompt = False
            for file_triplet in json_dict["data_files"]:
                file_path, modality, file_type = file_triplet.strip().split(",")

                # the query result of spk_prompt is speaker-id, not np.array
                # skip passing it to IterableDataset
                if modality == "spk":
                    use_speaker_prompt = True
                    continue

                # use the stem file name as the name
                iterator_path_name_type_list.append(
                    (
                        file_path,
                        file_path.split("/")[-1],
                        file_type,
                    )
                )

            key_list = json_dict["examples"]
            if key_dict is not None:
                key_list = [
                    key for key in key_list if json_dict["task"] + "_" + key in key_dict
                ]

            iterator = IterableESPnetDataset(
                path_name_type_list=iterator_path_name_type_list,
                preprocess=preprocess,
                key_file=key_list,
                preprocess_prefix=json_dict["task"] + " ",
                **kwargs,
            )
            self.data_iterators.append(iterator)
            self.task_map[iterator] = json_dict["task"]

            if use_speaker_prompt:
                self.speaker_prompt_config[iterator] = json_dict[
                    "speaker_prompt_length"
                ]

        # Keep same interface with IterableDataset
        self.apply_utt2category = False

        self.encoder_decoder_format = getattr(
            preprocess, "encoder_decoder_format", False
        )

    def __iter__(self):
        # (Jinchun): always add task as prefix, as one dataset can
        # be used in multiple tasks.
        for iterator in self.data_iterators:
            for uid, data in iterator:
                print(
                    "get an example: ", uid, data, self.task_map[iterator], flush=True
                )
                uid = self.task_map[iterator] + "_" + uid
                data = self.post_process(data, iterator)
                yield uid, data

    # Keep same interface with IterableDataset
    def has_name(self, name) -> bool:
        return name in self.names()

    def names(self) -> Tuple[str, ...]:
        if self.encoder_decoder_format:
            return ("encoder_sequence", "decoder_sequence")
        else:
            return "decoder_sequence"

    def __repr__(self):
        string = "##### Multi-Task Dataset #####\n"
        for idx, (dataset, task) in enumerate(self.task_map.items()):
            string += f"## Sub-Dataset: {idx}, Task: {task} ##\n"
            string += f"{dataset}\n"
        return string

    def post_process(self, data: Dict, iterator: IterableESPnetDataset):
        # (1) speaker prompt: dummy prompt, but with the correct length
        spk_conf = self.speaker_prompt_config[iterator]
        for key in data.keys():
            if key in spk_conf["names"]:
                break
        data[key] = np.ones(spk_conf["length"]).astype(np.int32)

        return data
