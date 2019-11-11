import collections
import logging
from io import StringIO
from typing import Dict, Mapping, List, Tuple, Union

import kaldiio
import numpy as np
import torch
from typeguard import typechecked
from torch.utils.data import Sampler

from espnet.nets.pytorch_backend.nets_utils import pad_list, make_pad_mask
from espnet.transform.transformation import Transformation
from espnet2.utils.fileio import SoundScpReader, scp2dict


class BatchSampler(Sampler):
    @typechecked
    def __init__(self,
                 batch_size: int,
                 type: str,
                 paths: Union[Tuple[str, ...], List[str]],
                 shuffle: bool = False):
        if len(paths) == 0:
            raise ValueError('1 or more paths must be given')
        paths = tuple(paths)

        self.shuffle = shuffle
        self.batch_size = batch_size

        if type == 'const':
            utt2length = scp2dict(paths[0])
            utt2length = {k: int(v) for k, v in utt2length.items()}
            # Sorted in descending order
            keys = sorted(utt2length, key=lambda k: -utt2length[k])

            self.batch_list = \
                [keys[i:i + batch_size]
                 for i in range(0, int(np.ceil(len(keys) / batch_size)),
                                batch_size)]

        # conventional behaviour of batchify()
        elif type == 'seq':
            raise NotImplementedError

        elif type == 'batchbin':
            raise NotImplementedError

        if self.shuffle:
            np.random.shuffle(self.batch_list)

    def __len__(self):
        raise len(self.batch_list)

    def __iter__(self):
        for batch in self.batch_list:
            yield batch


@typechecked
def collate_fn(data: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Concat ndarray-list and convert to torch.Tensor.

    Examples:
        Simple data flow from data-creation to DNN-forward

        >>> sampler = BatchSampler(...)
        >>> dataset = Dataset(...)

        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert all(set(data[0]) == set(d) for d in data), 'dict-keys mismatching'
    assert all(k + 'lengths' not in data[0] for k in data[0]), \
        '*_lengths is reserved: {list(data[0})'

    output = {}
    for key in data[0]:
        # Note(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == 'f':
            pad_value = -np.inf
        elif data[0][key].dtype == np.bool:
            pad_value = 0
        else:
            pad_value = -32768

        array_list = [d[key] for d in data]
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
    @typechecked
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
        return array


class Dataset:
    """

    Examples:
        >>> dataset = Dataset(dict(input=dict(path='wav.scp', type='sound'),
        ...                        output=dict(path='token_int',
        ...                                    type='csv_int')),
        ...                   dict(input=[dict(type='fbank',
        ...                                    n_mels=80, fs=16000)]))
        ... data = dataset['uttid']
        {'input': ndarray, 'output': ndarray}
    """

    @typechecked
    def __init__(self, config: dict, preproces: dict = None,
                 float_dtype: str = 'float32', int_dtype: str = 'long'):
        if len(config) == 0:
            raise ValueError('1 or more elements are required for "config"')
        config = config.copy()
        preproces = preproces.copy()

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        self.loader_dict = {}
        self.debug_info = {}
        for key, data in config.items():
            path = data['path']
            _type = data['type']

            loader = Dataset.create_loader(path, _type)
            self.loader_dict[key] = loader
            self.debug_info[key] = path, _type

        self.preprocess_dict = {}
        if preproces is not None:
            for key, data in preproces.items():
                proceess = Transformation(data)
                self.preprocess_dict[key] = proceess

            # The keys of preprocess must be sub-set of the keys of dataset
            for k in self.preprocess_dict:
                if k not in self.loader_dict:
                    raise RuntimeError(
                        f'The preprocess-key doesn\'t exit in data-keys: '
                        f'{k} not in {set(self.loader_dict)}')

    def __len__(self):
        raise RuntimeError(
            'Not necessary to be used because '
            'we are using custom batch-sampler')

    @staticmethod
    @typechecked
    def create_loader(path: str, loader_type: str) -> Mapping[str, np.ndarray]:
        if loader_type == 'sound':
            # path looks like:
            #   utta /some/where/a.wav
            #   uttb /some/where/a.flac
            #
            # Note that SoundScpReader doesn't support pipe-fashion
            # like Kaldi e.g. "cat a.wav |".
            loader = SoundScpReader(path, normalize=True)

            # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
            # but ndarray is desired, so Adapter class is inserted here
            return AdapterForSoundScpReader(loader)

        elif loader_type == 'ark_scp':
            # path looks like:
            #   utta /some/where/a.ark:123
            #   uttb /some/where/a.ark:456
            return kaldiio.load_scp(path)

        elif loader_type == 'npy_scp':
            # path looks like:
            #   utta /some/where/a.npy
            #   uttb /some/where/b.npy
            raise NotImplementedError

        elif loader_type in ('text_int', 'text_float', 'csv_int', 'csv_float'):
            if loader_type == 'text_int':
                delimiter = ' '
                dtype = np.long
            elif loader_type == 'text_float':
                delimiter = ' '
                dtype = np.float32
            elif loader_type == 'csv_int':
                delimiter = ','
                dtype = np.long
            elif loader_type == 'csv_float':
                delimiter = ','
                dtype = np.float32
            else:
                raise RuntimeError('Can\'t reach')

            # path looks like:
            #   utta 1,0
            #   uttb 3,4,5
            # -> return {'utta': np.ndarray([1, 0]),
            #            'uttb': np.ndarray([3, 4, 5])}
            d = scp2dict(path)

            # Using for-loop instead of dict-comprehension for debuggability
            retval = {}
            for k, v in d.items():
                try:
                    retval[k] = np.loadtxt(
                        StringIO(v), ndmin=1, dtype=dtype, delimiter=delimiter)
                except Exception:
                    logging.error(f'Error happened with path="{path}", '
                                  f'id="{k}", value="{v}"')
                    raise
            return retval

        else:
            raise RuntimeError(
                f'Not supported: loader_type={loader_type}')

    @typechecked
    def __getitem__(self, uid: str) -> Dict[str, np.ndarray]:
        data = {}
        for name, loader in self.loader_dict.items():
            try:
                value = loader[uid]
                assert isinstance(value, np.ndarray), type(value)
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
                    f'Not supported dtype: {value.kind}')

            data[name] = value

        return data

