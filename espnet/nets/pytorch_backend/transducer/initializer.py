#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parameter initialization for transducer model."""

import math

from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one


def initializer(model, args):
    """Initialize transducer model.

    Args:
        model (torch.nn.Module): transducer instance
        args (Namespace): argument Namespace containing options

    """
    for name, p in model.named_parameters():
        if any(x in name for x in ["enc.", "dec.", "joint_network"]):
            # rnn based parts + joint network
            if p.dim() == 1:
                # bias
                p.data.zero_()
            elif p.dim() == 2:
                # linear weight
                n = p.size(1)
                stdv = 1.0 / math.sqrt(n)
                p.data.normal_(0, stdv)
            elif p.dim() in (3, 4):
                # conv weight
                n = p.size(1)
                for k in p.size()[2:]:
                    n *= k
                    stdv = 1.0 / math.sqrt(n)
                    p.data.normal_(0, stdv)

    if args.dtype != "custom":
        model.dec.embed.weight.data.normal_(0, 1)

        for i in range(model.dec.dlayers):
            set_forget_bias_to_one(getattr(model.dec.decoder[i], "bias_ih_l0"))
            set_forget_bias_to_one(getattr(model.dec.decoder[i], "bias_hh_l0"))
