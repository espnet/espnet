# Copyright 2020 The ESPnet Authors.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""V1 backend for `asr_recog.py`."""

from itertools import zip_longest
import json
import logging

import numpy as np
import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.lm.pytorch_backend import extlm
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.lm import default
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    if args.streaming_mode and "transformer" in train_args.model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")
    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError(
                "use '--api v2' option to decode with non-default language model"
            )
        rnnlm = default.ClassifierWithState(
            default.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = default.ClassifierWithState(
            default.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = default.ClassifierWithState(
                extlm.MultiLevelLM(
                    word_rnnlm.predictor, rnnlm.predictor, word_dict, char_dict
                )
            )
        else:
            rnnlm = default.ClassifierWithState(
                extlm.LookAheadWordLM(word_rnnlm.predictor, word_dict, char_dict)
            )

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = (
                    feat[0][0]
                    if args.num_encs == 1
                    else [feat[idx][0] for idx in range(model.num_encs)]
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with window size %d frames",
                        args.streaming_window,
                    )
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info(
                            "Feeding frames %d - %d", i, i + args.streaming_window
                        )
                        se2e.accept_input(feat[i : i + args.streaming_window])
                    logging.info("Running offline attention decoder")
                    se2e.decode_with_attention_offline()
                    logging.info("Offline attention decoder finished")
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with threshold value %d",
                        args.streaming_min_blank_dur,
                    )
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                elif hasattr(model, "decoder_mode") and model.decoder_mode == "maskctc":
                    nbest_hyps = model.recognize_maskctc(
                        feat, args, train_args.char_list
                    )
                else:
                    nbest_hyps = model.recognize(
                        feat, args, train_args.char_list, rnnlm
                    )
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )

    else:

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = (
                    load_inputs_and_targets(batch)[0]
                    if args.num_encs == 1
                    else load_inputs_and_targets(batch)
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(
                        feats, args, train_args.char_list, rnnlm=rnnlm
                    )

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyp, train_args.char_list
                    )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
