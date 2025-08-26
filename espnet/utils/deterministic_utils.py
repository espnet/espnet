"""Deterministic Utils methods."""

import logging
import os

import chainer
import torch


def set_deterministic_pytorch(args):
    """Ensure pytorch produces deterministic results depending on program arguments.

    :param Namespace args: The program arguments
    """
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # considering reproducibility
    # remove type check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # https://github.com/pytorch/pytorch/issues/6351
    )
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info("torch type check is disabled")
    # use deterministic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logging.info("torch cudnn deterministic is disabled")


def set_deterministic_chainer(args):
    """Ensure chainer produces deterministic results depending on program arguments.

    :param Namespace args: The program arguments
    """
    # seed setting (chainer seed may not need it)
    os.environ["CHAINER_SEED"] = str(args.seed)
    logging.info("chainer seed = " + os.environ["CHAINER_SEED"])

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # considering reproducibility
    # remove type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info("chainer type check is disabled")
    # use deterministic computation or not
    if args.debugmode < 1:
        chainer.config.cudnn_deterministic = False
        logging.info("chainer cudnn deterministic is disabled")
    else:
        chainer.config.cudnn_deterministic = True
