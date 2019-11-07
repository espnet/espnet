from typing import Dict, Mapping, List

import kaldiio
import numpy as np
from pytypes import typechecked
import torch
from torch.utils.data import Sampler

from espnet.nets.pytorch_backend.nets_utils import pad_list, make_pad_mask
from espnet.transform.transformation import Transformation
from espnet2.utils.fileio import SoundScpReader, scp2dict


class BatchSampler(Sampler):
    @typechecked
    def __init__(self, config: dict, shuffle: bool = False):
        """

        config: e.g.

        type: seq
        shape:
            - utt2shape
        batch_size: 10
        """
        self.shuffle = shuffle
        batch_size = config['batch_size']
        self.batch_size = batch_size

        if config['type'] == 'const':
            path = config['path']
            utt2length = scp2dict(path)
            utt2length = {k: int(v) for k, v in utt2length}
            # Sorted in descending order
            keys = sorted(utt2length, key=lambda k: -utt2length[k])

            self.batch_list = \
                [keys[i:i + batch_size]
                 for i in range(0, len(keys) // 2 + 1, batch_size)]

        # conventional behaviour of batchify()
        elif config['type'] == 'seq':
            raise NotImplementedError
        elif config['type'] == 'batchbin':
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
    """

    Concat ndarray-list and convert to torch.Tensor.

    Examples:
        Simple data flow from data-creation to DNN-forward

        >>> sampler = BatchSampler(...)
        >>> dataset = Dataset(...)

        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch is propagated from
        that of the dataset,
        and they can be changed

    """
    assert all(set(data[0]) == set(d) for d in data), 'dict-keys mismatching'

    output = {}
    for key in data[0]:
        # Note(kamo):
        # Eaach models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == 'f':
            pad_value = -np.inf
        elif data[0][key].dtype == np.bool:
            pad_value = 0
        else:
            pad_value = -32768

        # tensor: (Batch, Length, ...)
        tensor = pad_list([d[key] for d in data], pad_value)
        output[key] = tensor

        # FIXME(kamo): I'm not sure which is better, mask or lengths.
        # mask: (Batch,)
        mask = make_pad_mask([len(d[key]) for d in data])
        output[key + '_mask'] = mask

    return output


class Dataset:
    @typechecked
    def __init__(self, config: dict):
        """

        config: e.g.

        data:
            input:
                  path: /some/where/wav.scp
                  type: sound
            output:
                  path: /some/where/utt2tokenid
                  type: text_int
        preprocess:
            input:
                - type: fbank
                  nfft: 512
                  window_length: 400
                  window_shift: 160

        """
        self.loader_dict = {}
        assert isinstance(config['data'], dict), config['data']
        assert isinstance(config['preprocess'], dict), config['preprocess']
        for key, data in config['data'].items():
            path = data['path']
            type = data['type']
            loader = Dataset.create_loader(path, type)
            self.loader_dict[key] = loader

        self.preprocess_dict = {}
        for key, data in config['preprocess'].items():
            proceess = Transformation(data)
            self.preprocess_dict[key] = proceess

    def __len__(self):
        raise RuntimeError(
            'Not necessary to be used because '
            'we are using custom batchーsampler')

    @staticmethod
    def create_loader(path: str, loader_type: str) -> Mapping[str, np.ndarray]:
        if loader_type == 'sound':
            return SoundScpReader(path)
        elif loader_type == 'ark-scp':
            return kaldiio.load_scp(path)
        elif loader_type == 'text_int':
            return {k: np.loadtxt(v, ndmin=1, dtype=np.long)
                    for k, v in scp2dict(path)}

        else:
            raise NotImplementedError(
                f'Not supported: loader_type={loader_type}')

    @typechecked
    def __getitem__(self, uid: str) -> Dict[str, np.ndarray]:
        data = {}
        for name, loader in self.loader_dict.items():
            value = loader[uid]
            if name in self.preprocess_dict:
                process = self.preprocess_dict[name]
                value = process(value)
            data[name] = value

        return data

