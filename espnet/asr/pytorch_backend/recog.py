import json
import logging

import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.beam_search import beam_search
from espnet.nets.pytorch_backend.beam_search import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets


def recog_v2(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.
    """
    logging.warning("experimental API for custom LMs is selected by --lm option")
    if args.batchsize > 0:
        raise NotImplementedError("batch decoding is not implemented")
    if args.ngpu > 0:
        raise NotImplementedError("GPU decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    lm = None
    if args.lm:
        lm_args = get_model_conf(args.lm, args.lm_conf)
        lm_class = dynamic_import_lm(lm_args.model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.lm, lm)
        lm.eval()

    decoders = [model.get_decoder(), model.get_ctc(), lm, LengthBonus()]
    weights = [1.0, args.ctc_weight, args.lm_weight, args.penalty]

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = model.encode(load_inputs_and_targets(batch)[0][0])
            nbest_hyps = beam_search(
                x=feat,
                beam_size=args.beam_size,
                weights=weights,
                decoders=decoders,
                token_list=train_args.char_list
            )
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
