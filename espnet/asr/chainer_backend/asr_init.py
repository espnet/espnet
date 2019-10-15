"""Finetuning methods."""
import logging
import os

from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import get_model_conf

from espnet.utils.dynamic_import import dynamic_import


def load_trained_model(model_path):
    """Load the trained model.

    Args:
        model_path(str): Path to model.***.best

    """
    # read training config
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), 'model.json'))

    # specify model architecture
    logging.info('reading model parameters from ' + model_path)
    # To be compatible with v.0.3.0 models
    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.chainer_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)
    chainer_load(model_path, model)

    return model, train_args
