# Copyright 2020 The ESPnet Authors.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""ASR converter module."""
import numpy as np
import torch

from espnet.nets.pytorch_backend.e2e_asr import pad_list


class CustomConverter:
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[:: self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
                device, dtype=self.dtype
            )

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(
                    np.array(y[0][:]) if isinstance(y, tuple) else y
                ).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_pad, ilens, ys_pad


class CustomConverterMulEnc(object):
    """Custom batch converter for Pytorch in multi-encoder case.

    Args:
        subsampling_factors (list): List of subsampling factors for each encoder.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsamping_factors=[1, 1], dtype=torch.float32):
        """Initialize the converter."""
        self.subsamping_factors = subsamping_factors
        self.ignore_id = -1
        self.dtype = dtype
        self.num_encs = len(subsamping_factors)

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple( list(torch.Tensor), list(torch.Tensor), torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs_list = batch[0][: self.num_encs]
        ys = batch[0][-1]

        # perform subsampling
        if np.sum(self.subsamping_factors) > self.num_encs:
            xs_list = [
                [x[:: self.subsampling_factors[i], :] for x in xs_list[i]]
                for i in range(self.num_encs)
            ]

        # get batch of lengths of input sequences
        ilens_list = [
            np.array([x.shape[0] for x in xs_list[i]]) for i in range(self.num_encs)
        ]

        # perform padding and convert to tensor
        # currently only support real number
        xs_list_pad = [
            pad_list([torch.from_numpy(x).float() for x in xs_list[i]], 0).to(
                device, dtype=self.dtype
            )
            for i in range(self.num_encs)
        ]

        ilens_list = [
            torch.from_numpy(ilens_list[i]).to(device) for i in range(self.num_encs)
        ]
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_list_pad, ilens_list, ys_pad
