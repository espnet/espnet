"""Initialize ESPnet2 Transducer model modules."""

import math
import torch
from typeguard import check_argument_types


def initialize(model: torch.nn.Module, init: str):
    """Initialize weights of a Transducer model.

    Args:
        model: Transducer model.
        init: Initialization method.

    """
    assert check_argument_types()

    if "chainer" in init:
        # 1. lecun_normal_init_parameters
        for p in model.parameters():
            data = p.data

            # bias init
            if p.dim() == 1:
                p.data.zero_()
            # linear weight
            elif data.dim() == 2:
                n = data.size(1)
                stdv = 1.0 / math.sqrt(n)

                data.normal_(0, stdv)
            # conv weight
            elif data.dim() in (3, 4):
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1.0 / math.sqrt(n)

                data.normal_(0, stdv)

        for name_mod, mod in model.named_modules():
            # 2. embed weight ~ Normal(0, 1)
            if isinstance(mod, torch.nn.Embedding):
                mod.weight.data.normal_(0, 1)
            # 3. forget-bias = 1.0
            elif isinstance(mod, torch.nn.RNNBase):
                # set decoder RNN forget bias only (ESPnet1 style)
                if init == "chainer_espnet1" and "encoder" in name_mod:
                    continue

                for name, param in mod.named_parameters():
                    if "bias" in name:
                        n = param.size(0)

                        param.data[n // 4 : n // 2].fill_(1.0)
    else:
        # weight init
        for p in model.parameters():
            if p.dim() == 1:
                p.data.zero_()
            elif p.dim() > 1:
                if init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + init)

        # reset some modules with default init
        for m in model.modules():
            if isinstance(
                m, (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.GroupNorm)
            ):
                m.reset_parameters()
