#!/usr/bin/env python3

"""Initialize modules for espnet2 neural networks."""

import logging

import torch
from typeguard import typechecked


@typechecked
def initialize(model: torch.nn.Module, init: str):
    """Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Custom initialization routines can be implemented into submodules
    as function `espnet_initialization_fn` within the custom module.

    Args:
        model: Target.
        init: Method of initialization.
    """
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
            elif init == "normal":
                torch.nn.init.normal_(p.data, mean=0.0, std=0.02)
            else:
                raise ValueError("Unknown initialization: " + init)
    # bias init
    for name, p in model.named_parameters():
        if ".bias" in name and p.dim() == 1:
            p.data.zero_()
            logging.info(f"Initialize {name} to zeros")

    # reset some modules with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            m.reset_parameters()
        if hasattr(m, "espnet_initialization_fn"):
            m.espnet_initialization_fn()

    # TODO(xkc): Hacking s3prl_frontend and wav2vec2encoder initialization
    if getattr(model, "encoder", None) and getattr(
        model.encoder, "reload_pretrained_parameters", None
    ):
        model.encoder.reload_pretrained_parameters()
    if getattr(model, "frontend", None):
        if getattr(model.frontend, "reload_pretrained_parameters", None):
            model.frontend.reload_pretrained_parameters()
        elif isinstance(
            getattr(model.frontend, "frontends", None),
            torch.nn.ModuleList,
        ):
            for i, _ in enumerate(getattr(model.frontend, "frontends")):
                if getattr(
                    model.frontend.frontends[i],
                    "reload_pretrained_parameters",
                    None,
                ):
                    model.frontend.frontends[i].reload_pretrained_parameters()
    if getattr(model, "postencoder", None) and getattr(
        model.postencoder, "reload_pretrained_parameters", None
    ):
        model.postencoder.reload_pretrained_parameters()
    if getattr(model, "decoder", None) and getattr(
        model.decoder, "reload_pretrained_parameters", None
    ):
        model.decoder.reload_pretrained_parameters()
