import json
import logging

import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search import beam_search
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets


def recog_v2(args):
    """New asr_recog.py backend to decode with custom models that implements ScorerInterface.

    Notes: the previous backend espnet.asr.pytorch_backend.asr.recog only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments. See espnet.bin.asr_recog.get_parser for details
    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    lm = None
    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        lm_class = dynamic_import_lm(lm_args.model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.rnnlm, lm)
        lm.eval()

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        length_bonus=args.penalty)
    if args.ngpu > 0:
        if args.ngpu > 1:
            raise NotImplementedError("only single GPU decoding is supported")
        logging.info("enable GPU decoding")
        model.cuda()
        for k, s in scorers.items():
            if isinstance(s, torch.nn.Module):
                s.cuda()
        device = "cuda"
    else:
        device = "cpu"

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            enc = model.encode(torch.as_tensor(feat).to(device))
            nbest_hyps = beam_search(
                x=enc,
                sos=model.sos,
                eos=model.eos,
                beam_size=args.beam_size,
                weights=weights,
                decoders=scorers,
                token_list=train_args.char_list,
                maxlenratio=args.maxlenratio,
                minlenratio=args.minlenratio)
            nbest_hyps = nbest_hyps[:min(len(nbest_hyps), args.nbest)]
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
