#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", type=str, nargs="+", help="json files")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--output-json", default="", type=str, help="output json file")
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    # make intersection set for utterance keys
    js = []
    intersec_ks = []
    for x in args.jsons:
        with open(x, "r") as f:
            j = json.load(f)
        ks = j["utts"].keys()
        logging.info(x + ": has " + str(len(ks)) + " utterances")
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info("new json has " + str(len(intersec_ks)) + " utterances")

    old_dic = dict()
    for k in intersec_ks:
        v = js[0]["utts"][k]
        for j in js[1:]:
            v.update(j["utts"][k])
        old_dic[k] = v

    new_dic = dict()
    for id in old_dic:
        dic = old_dic[id]

        in_dic = {}
        # if unicode('idim', 'utf-8') in dic:
        if "idim" in dic:
            in_dic["shape"] = (
                int(dic["ilen"]),
                int(dic["idim"]),
            )
        in_dic["name"] = "input1"
        in_dic["feat"] = dic["feat"]

        out_list = []
        out_idx = 1
        while f"text_spk{out_idx}" in dic:
            out_dic = {}
            out_dic["name"] = f"target{out_idx}"
            out_dic["shape"] = int(dic[f"olen_spk{out_idx}"]), int(dic["odim"])
            out_dic["text"] = dic[f"text_spk{out_idx}"]
            out_dic["token"] = dic[f"token_spk{out_idx}"]
            out_dic["tokenid"] = dic[f"tokenid_spk{out_idx}"]
            out_list.append(out_dic)
            out_idx += 1

        new_dic[id] = {
            "input": [in_dic],
            "output": out_list,
            "utt2spk": dic["utt2spk"],
        }

    # ensure "ensure_ascii=False", which is a bug
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as json_file:
            json.dumps(
                {"utts": new_dic},
                json_file,
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
                encoding="utf-8",
            )
    else:
        sys.stdout = codecs.getwriter("utf8")(sys.stdout)
        json.dump(
            {"utts": new_dic},
            sys.stdout,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
            encoding="utf-8",
        )
