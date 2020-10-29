#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import codecs
import json
import logging
import os
import sys

from espnet.utils.cli_utils import get_commandline_args

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description="merge json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-jsons",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="Json files for the inputs",
    )
    parser.add_argument(
        "--output-jsons",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="Json files for the outputs",
    )
    parser.add_argument(
        "--jsons",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="The json files except for the input and outputs",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("-O", dest="output", type=str, help="Output json file")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    js_dict = {}  # Dict[str, List[List[Dict[str, Dict[str, dict]]]]]
    # make intersection set for utterance keys
    intersec_ks = None  # Set[str]
    for jtype, jsons_list in [
        ("input", args.input_jsons),
        ("output", args.output_jsons),
        ("other", args.jsons),
    ]:
        js_dict[jtype] = []
        for jsons in jsons_list:
            js = []
            for x in jsons:
                if os.path.isfile(x):
                    with codecs.open(x, encoding="utf-8") as f:
                        j = json.load(f)
                    ks = list(j["utts"].keys())
                    logging.info(x + ": has " + str(len(ks)) + " utterances")
                    if intersec_ks is not None:
                        intersec_ks = intersec_ks.intersection(set(ks))
                        if len(intersec_ks) == 0:
                            logging.warning("No intersection")
                            break
                    else:
                        intersec_ks = set(ks)
                    js.append(j)
            js_dict[jtype].append(js)
    logging.info("new json has " + str(len(intersec_ks)) + " utterances")

    new_dic = {}
    for k in intersec_ks:
        new_dic[k] = {"input": [], "output": []}
        for jtype in ["input", "output", "other"]:
            for idx, js in enumerate(js_dict[jtype], 1):
                # Merge dicts from jsons into a dict
                dic = {k2: v for j in js for k2, v in j["utts"][k].items()}

                if jtype == "other":
                    new_dic[k].update(dic)
                else:
                    _dic = {}

                    # FIXME(kamo): ad-hoc way to change str to List[int]
                    if jtype == "input":
                        _dic["name"] = "input{}".format(idx)
                        if "ilen" in dic and "idim" in dic:
                            _dic["shape"] = (int(dic["ilen"]), int(dic["idim"]))
                        elif "ilen" in dic:
                            _dic["shape"] = (int(dic["ilen"]),)
                        elif "idim" in dic:
                            _dic["shape"] = (int(dic["idim"]),)

                    elif jtype == "output":
                        _dic["name"] = "target{}".format(idx)
                        if "olen" in dic and "odim" in dic:
                            _dic["shape"] = (int(dic["olen"]), int(dic["odim"]))
                        elif "ilen" in dic:
                            _dic["shape"] = (int(dic["olen"]),)
                        elif "idim" in dic:
                            _dic["shape"] = (int(dic["odim"]),)
                    if "shape" in dic:
                        # shape: "80,1000" -> [80, 1000]
                        _dic["shape"] = list(map(int, dic["shape"].split(",")))

                    for k2, v in dic.items():
                        if k2 not in ["ilen", "idim", "olen", "odim", "shape"]:
                            _dic[k2] = v
                    new_dic[k][jtype].append(_dic)

    # ensure "ensure_ascii=False", which is a bug
    if args.output is not None:
        sys.stdout = codecs.open(args.output, "w", encoding="utf-8")
    else:
        sys.stdout = codecs.getwriter("utf-8")(
            sys.stdout if is_python2 else sys.stdout.buffer
        )
    print(
        json.dumps(
            {"utts": new_dic},
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "),
        )
    )
