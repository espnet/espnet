import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.transformer.initializer import initialize as \
    transformer_initialize


def initialize(model: torch.nn.Module, init: str):
    assert check_argument_types()
    if init == 'chainer':
        # embed weight ~ Normal(0, 1)
        for mod in model.modules():
                # embed weight ~ Normal(0, 1)
            if isinstance(mod, torch.nn.Embedding):
                mod.weight.data.normal_(0, 1)
            elif isinstance(mod, torch.nn.RNNCellBase):
                # forget-bias = 1.0
                # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
                set_forget_bias_to_one(model.bias_ih)
    else:
        transformer_initialize(model, init)
