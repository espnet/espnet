#!/usr/bin/env python3

# Copyright 2020 Shanghai Jiao Tong University (Wangyou Zhang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import sys
from functools import reduce
from operator import mul

from espnet.bin.asr_train import get_parser
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.utils.dynamic_import import dynamic_import

if __name__ == "__main__":
    cmd_args = sys.argv[1:]
    parser = get_parser(required=False)
    parser.add_argument("--data-json", type=str, help="data.json")
    parser.add_argument(
        "--mode-subsample", type=str, required=True, help='One of ("asr", "mt", "st")'
    )
    parser.add_argument(
        "--min-io-delta",
        type=float,
        help="An additional parameter "
        "for controlling the input-output length difference",
        default=0.0,
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        required=True,
        help="Output path of the filtered json file",
    )
    args, _ = parser.parse_known_args(cmd_args)

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module

    module_name = model_module.split(":")[0].split(".")[-1]
    # One of ("rnn", "rnn-t", "rnn_mix", "rnn_mulenc", "transformer")
    if module_name == "e2e_asr":
        arch_subsample = "rnn"
    elif module_name == "e2e_asr_transducer":
        arch_subsample = "rnn-t"
    elif module_name == "e2e_asr_mix":
        arch_subsample = "rnn_mix"
    elif module_name == "e2e_asr_mulenc":
        arch_subsample = "rnn_mulenc"
    elif "transformer" in module_name:
        arch_subsample = "transformer"
    else:
        raise ValueError("Unsupported model module: %s" % model_module)

    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    # subsampling info
    if hasattr(args, "etype") and args.etype.startswith("vgg"):
        # Subsampling is not performed for vgg*.
        # It is performed in max pooling layers at CNN.
        min_io_ratio = 4
    else:
        subsample = get_subsample(args, mode=args.mode_subsample, arch=arch_subsample)
        # the minimum input-output length ratio for all samples
        min_io_ratio = reduce(mul, subsample)

    # load dictionary
    with open(args.data_json, "rb") as f:
        j = json.load(f)["utts"]

    # remove samples with IO ratio smaller than `min_io_ratio`
    for key in list(j.keys()):
        ilen = j[key]["input"][0]["shape"][0]
        olen = min(x["shape"][0] for x in j[key]["output"])
        if float(ilen) - float(olen) * min_io_ratio < args.min_io_delta:
            j.pop(key)
            print("'{}' removed".format(key))

    jsonstring = json.dumps({"utts": j}, indent=4, ensure_ascii=False, sort_keys=True)
    with open(args.output_json_path, "w") as f:
        f.write(jsonstring)
