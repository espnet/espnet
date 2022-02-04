"""Initialize ESPnet2 Transducer model modules."""

import math
import torch
from typeguard import check_argument_types


def initialize(model: torch.nn.Module):
    """Initialize weights of a Transducer model.

    Args:
        model: Transducer model.

    """
    assert check_argument_types()

    # 1. lecun_normal_init_parameters
    for p in model.parameters():
        data = p.data

        # bias
        if data.dim() == 1:
            data.zero_()
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
        else:
            raise NotImplementedError

    for name_mod, mod in model.named_modules():
        # 2. embed weight ~ Normal(0, 1)
        if isinstance(mod, torch.nn.Embedding):
            mod.weight.data.normal_(0, 1)
        # 3. forget-bias = 1.0
        elif "encoder" not in name_mod and isinstance(mod, torch.nn.RNNBase):
            for name, param in mod.named_parameters():
                if "bias" in name:
                    n = param.size(0)

                    param.data[n // 4 : n // 2].fill_(1.0)
