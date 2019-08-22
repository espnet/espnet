"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import chainer

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import chainer_load
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

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        lm_class = dynamic_import_lm(lm_args.model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        chainer_load(args.rnnlm, lm)
    else:
        lm = None
    
    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        length_bonus=args.penalty)
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(train_args.char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=train_args.char_list,
    )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"