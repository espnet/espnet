import collections
import copy
import logging
from typing import Dict, Mapping, Sequence, Union

import kaldiio
import numpy as np
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.transform.transformation import Transformation
from espnet2.utils.fileio import SoundScpReader, load_num_sequence_text


@typechecked
def our_collate_fn(data: Sequence[Dict[str, np.ndarray]]) \
        -> Dict[str, torch.Tensor]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.train.batch_sampler import ConstantBatchSampler
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPNetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = our_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert all(set(data[0]) == set(d) for d in data), 'dict-keys mismatching'
    assert all(k + '_lengths' not in data[0] for k in data[0]), \
        '*_lengths is reserved: {list(data[0})'

    output = {}
    for key in data[0]:
        # Note(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == 'f':
            pad_value = -np.inf
        else:
            pad_value = -32768

        array_list = [d[key] for d in data]
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        assert all(len(d[key]) != 0 for d in data), [len(d[key]) for d in data]

        # lens: (Batch,)
        lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
        output[key + '_lengths'] = lens

    return output


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(self, loader: SoundScpReader):
        self.loader = loader
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
            raise RuntimeError(
                f'Sampling rates are mismatched: {self.rate} != {rate}')
        self.rate = rate
        # Multichannel wave fie
        if array.ndim == 2:
            # (NSample, Channel) -> (Channel, NSample)
            array = array.T
        return array


class ESPNetDataset:
    """

    Examples:
        >>> dataset = ESPNetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                          preproces=dict(input=[dict(type='fbank',
        ...                                         n_mels=80, fs=16000)])
        ...                         )
        ... data = dataset['uttid']
        {'input': ndarray, 'output': ndarray}
    """

    @typechecked
    def __init__(self, path_name_type_list: Sequence[Sequence[str]],
                 preproces: Dict[str, Union[str, dict]] = None,
                 float_dtype: str = 'float32', int_dtype: str = 'long'):
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"')
        # str is also Sequence[Sequence[str]]
        if isinstance(path_name_type_list, str):
            raise TypeError('config must be Sequence[Sequence[str]], '
                            'but got str')
        for _config in path_name_type_list:
            # str is also Sequence[str]
            if isinstance(_config, str):
                raise TypeError('config must be Sequence[Sequence[str]], '
                                'but got Sequence[str]')
            if len(_config) != 3:
                raise ValueError(f'Must be a sequence of 3 str: '
                                 f'path, name, and, type: {_config}')

        path_name_type_list = copy.deepcopy(path_name_type_list)
        preproces = copy.deepcopy(preproces)

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            loader = ESPNetDataset.create_loader(path, _type)
            self.loader_dict[name] = loader
            self.debug_info[name] = path, _type
            if len(self.loader_dict[name]) == 0:
                raise RuntimeError(f'{path} has no samples')
        self._uids = \
            tuple(set.intersection(*[set(d) for d in self.loader_dict]))

        logging.info(
            'The number of samples: ' +
            ' '.join(f'{n}={len(self.loader_dict[n])}, '
                     for _, n, _ in path_name_type_list) +
            f' -> Intersection={len(self._uids)}')
        if len(self._uids) == 0:
            raise RuntimeError('No samples remain: ' +
                               ', '.join(p for p, _, _ in path_name_type_list))

        self.preprocess_dict = {}
        if preproces is not None:
            for name, data in preproces.items():
                proceess = Transformation(data)
                self.preprocess_dict[name] = proceess

            # The keys of preprocess must be sub-set of the keys of dataset
            for name in self.preprocess_dict:
                if name not in self.loader_dict:
                    raise RuntimeError(
                        f'The preprocess-key doesn\'t exist in data-keys: '
                        f'{name} not in {set(self.loader_dict)}')

    @staticmethod
    @typechecked
    def create_loader(path: str, loader_type: str) -> Mapping[str, np.ndarray]:
        """Helper function to instantiate Loader

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """
        if loader_type == 'sound':
            # path looks like:
            #   utta /some/where/a.wav
            #   uttb /some/where/a.flac

            # Note(kamo): I recommend "flac" format for audio file
            # because "flac" is one of lossless compression format and
            # and it has not bad compression performance and
            # can be decoded quickly.

            # Note(kamo): SoundScpReader doesn't support pipe-fashion
            # like Kaldi e.g. "cat a.wav |".

            # Note(kamo): The audio signal is normalized to [-1,1] range.

            loader = SoundScpReader(path, normalize=True, always_2d=False)

            # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
            # but ndarray is desired, so Adapter class is inserted here
            return AdapterForSoundScpReader(loader)

        elif loader_type == 'pipe_wav':
            # path looks like:
            #   utta cat a.wav |
            #   uttb cat b.wav |

            # Note(kamo): I don't think this case is practical
            # because subprocess takes much times due to fork() system call.

            loader = kaldiio.load_scp(path)
            return AdapterForSoundScpReader(loader)

        elif loader_type == 'ark_scp':
            # path looks like:
            #   utta /some/where/a.ark:123
            #   uttb /some/where/a.ark:456
            return kaldiio.load_scp(path)

        elif loader_type == 'npy':
            # path looks like:
            #   utta /some/where/a.npy
            #   uttb /some/where/b.npy
            raise NotImplementedError

        elif loader_type in ('text_int', 'text_float', 'csv_int', 'csv_float'):
            # Not lazy loader, but as vanilla-dict
            return load_num_sequence_text(path, loader_type)

        else:
            raise RuntimeError(
                f'Not supported: loader_type={loader_type}')

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += f'(Nsamples: {len(self)}'
        for name, (path, _type) in self.debug_info.items():
            _mes += (f'\n{name}:'
                     f'    path: {path}\n'
                     f'    type: {_type}')
        for name, preproces in self.preprocess_dict.items():
            _mes += (f'\n{name}:'
                     f'    preprocess: {preproces}')
        _mes += ')'
        return _mes

    def __len__(self):
        return len(self._uids)

    def keys(self):
        return self._uids

    # Note(kamo):
    # Typically pytorch's Dataset.__getitem__ accepts an inger index,
    # however this Dataset handle a string, which represents a sample-id.
    @typechecked
    def __getitem__(self, uid: Union[str, int]) -> Dict[str, np.ndarray]:
        if isinstance(uid, int):
            uid = self._uids[uid]

        data = {}
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                if not isinstance(value, np.ndarray):
                    raise TypeError(f'Must be ndarray: {type(value)}')
            except Exception:
                path, _type = self.debug_info[name]
                logging.error(
                    f'Error happened with path={path}, type={_type}, id={uid}')
                raise

            if name in self.preprocess_dict:
                process = self.preprocess_dict[name]
                value = process(value)

            # Cast to desired type
            if value.dtype.kind == 'f':
                value = value.astype(self.float_dtype)
            elif value.dtype.kind == 'i':
                value = value.astype(self.int_dtype)
            else:
                raise NotImplementedError(
                    f'Not supported dtype: {value.dtype}')

            data[name] = value

        return data

