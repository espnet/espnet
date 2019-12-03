from typing import Sequence, Dict

import numpy as np
from typeguard import check_argument_types, check_return_type
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


def common_collate_fn(data: Sequence[Dict[str, np.ndarray]]) \
        -> Dict[str, torch.Tensor]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.train.batch_sampler import ConstantBatchSampler
        >>> from espnet2.train.dataset import ESPNetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPNetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    assert all(set(data[0]) == set(d) for d in data), 'dict-keys mismatching'
    assert all(k + '_lengths' not in data[0] for k in data[0]), \
        f'*_lengths is reserved: {list(data[0])}'

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == 'f':
            pad_value = 0.
        else:
            pad_value = -32768

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        assert all(len(d[key]) != 0 for d in data), [len(d[key]) for d in data]

        # lens: (Batch,)
        lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
        output[key + '_lengths'] = lens

    assert check_return_type(output)
    return output
