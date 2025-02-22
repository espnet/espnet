# -*- coding: utf-8 -*-

"""Fill missing args methods."""

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging


def fill_missing_args(args, add_arguments):
    """Fill missing arguments in args.

    Args:
        args (Namespace or None): Namesapce containing hyperparameters.
        add_arguments (function): Function to add arguments.

    Returns:
        Namespace: Arguments whose missing ones are filled with default value.

    Examples:
        >>> from argparse import Namespace
        >>> from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2
        >>> args = Namespace()
        >>> fill_missing_args(args, Tacotron2.add_arguments_fn)
        Namespace(aconv_chans=32, aconv_filts=15, adim=512, atype='location', ...)

    """
    # check argument type
    assert isinstance(args, argparse.Namespace) or args is None
    assert callable(add_arguments)

    # get default arguments
    default_args, _ = add_arguments(argparse.ArgumentParser()).parse_known_args()

    # convert to dict
    args = {} if args is None else vars(args)
    default_args = vars(default_args)

    for key, value in default_args.items():
        if key not in args:
            logging.info(
                'attribute "%s" does not exist. use default %s.' % (key, str(value))
            )
            args[key] = value

    return argparse.Namespace(**args)
