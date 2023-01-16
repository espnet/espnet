# noqa E501: Ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/src/models/adapt_layers.py
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from functools import partial

import torch
import torch.nn as nn


def make_adapt_layer(type, indim, enrolldim, ninputs=1):
    adapt_class = adaptation_layer_types.get(type)
    return adapt_class(indim, enrolldim, ninputs)


def into_tuple(x):
    """Transforms tensor/list/tuple into tuple."""
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return (x,)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError("x should be tensor, list of tuple")


def into_orig_type(x, orig_type):
    """Inverts into_tuple function."""
    if orig_type is tuple:
        return x
    if orig_type is list:
        return list(x)
    if orig_type is torch.Tensor:
        return x[0]
    else:
        assert False


class ConcatAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1):
        super().__init__()
        self.ninputs = ninputs
        self.transform = nn.ModuleList(
            [nn.Linear(indim + enrolldim, indim) for _ in range(ninputs)]
        )

    def forward(self, main, enroll):
        """ConcatAdaptLayer forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs

        out = []
        for transform, main0, enroll0 in zip(self.transform, main, enroll):
            out.append(
                transform(
                    torch.cat(
                        (main0, enroll0[:, :, None].expand(main0.shape)), dim=1
                    ).permute(0, 2, 1)
                ).permute(0, 2, 1)
            )
        return into_orig_type(tuple(out), orig_type)


class MulAddAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1, do_addition=True):
        super().__init__()
        self.ninputs = ninputs
        self.do_addition = do_addition

        if do_addition:
            assert enrolldim == 2 * indim, (enrolldim, indim)
        else:
            assert enrolldim == indim, (enrolldim, indim)

    def forward(self, main, enroll):
        """MulAddAdaptLayer Forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs, (
            len(main),
            len(enroll),
            self.ninputs,
        )

        out = []
        for main0, enroll0 in zip(main, enroll):
            if self.do_addition:
                enroll0_mul, enroll0_add = torch.chunk(enroll0, 2, dim=1)
                out.append(enroll0_mul[:, :, None] * main0 + enroll0_add[:, :, None])
            else:
                out.append(enroll0[:, :, None] * main0)
        return into_orig_type(tuple(out), orig_type)


# aliases for possible adaptation layer types
adaptation_layer_types = {
    "concat": ConcatAdaptLayer,
    "muladd": MulAddAdaptLayer,
    "mul": partial(MulAddAdaptLayer, do_addition=False),
}
