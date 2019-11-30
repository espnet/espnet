import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.initialization import \
    lecun_normal_init_parameters


def initialize(model: torch.nn.Module, init: str):
    assert check_argument_types()
    if init == 'chainer':
        lecun_normal_init_parameters(model)

        for mod in model.modules():
            if isinstance(mod, torch.nn.Embedding):
                # embed weight ~ Normal(0, 1)
                mod.weight.data.normal_(0, 1)
            elif isinstance(mod, torch.nn.RNNCellBase):
                # forget-bias = 1.0
                n = mod.bias_ih.size(0)
                mod.bias_ih.data[n // 4:n // 2].fill_(1.)
    else:
        # weight init
        for p in model.parameters():
            if p.dim() > 1:
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
        # bias init
        for p in model.parameters():
            if p.dim() == 1:
                p.data.zero_()

        # reset some modules with default init
        for m in model.modules():
            if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm)):
                m.reset_parameters()
