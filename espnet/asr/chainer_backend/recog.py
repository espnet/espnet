"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import chainer

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.chainer_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.io_utils import LoadInputsAndTargets


def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.chainer_backend.asr.recog only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments. See py:func:`espnet.bin.asr_recog.get_parser` for details

    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")
    
    set_deterministic_chainer(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()