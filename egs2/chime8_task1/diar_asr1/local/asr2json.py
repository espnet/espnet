import argparse
import json
import os
import re
from pathlib import Path


def parse2json(asr_hyp, output_name, sess_regex="[^-]*"):
    with open(asr_hyp, "r") as f:
        hyps = f.readlines()
    hyps = [x.rstrip("\n") for x in hyps]
    to_json = []
    # session, spk_id, start and stop in seconds and words
    for h in hyps:
        utt_id = h.split(" ")[0]
        predictions = " ".join(h.split(" ")[1:])
        spk_id = utt_id.split("_")[0]

        rest = "_".join(utt_id.split("_")[1:])
        # mixer6 regex: ([0-9]+_[0-9]+_LDC_[0-9]+)
        session_id = re.search(sess_regex, rest).group()
        start, stop = re.search("-([0-9]*_[0-9]*)", rest).group().lstrip("-").split("_")
        start = float(start) / 100
        stop = float(stop) / 100

        c_entry = {
            "words": predictions,
            "speaker": spk_id,
            "session_id": session_id,
            "start_time": str(start),
            "end_time": str(stop),
        }
        to_json.append(c_entry)

    # sort by starting time
    Path(output_name).mkdir(parents=True, exist_ok=True)
    to_json = sorted(
        to_json, key=lambda x: "{}_{}".format(x["speaker"], x["start_time"])
    )
    with open(output_name + ".json", "w") as f:
        json.dump(to_json, f, indent=4)

    # write also to stm for asclite

    stm = []
    for seg in to_json:
        c_line = "{} A {} {} {} {}\n".format(
            seg["session_id"],
            seg["speaker"],
            seg["start_time"],
            seg["end_time"],
            seg["words"],
        )
        stm.append(c_line)

    with open(output_name + ".stm", "w") as f:
        f.writelines(stm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script parses ESPNet2 asr.sh generated transcriptions and put them into "
        "a CHiME-7 DASR JSON file to enable meeting-long evaluation. "
        "It is assumed that the utterance id starts with the speaker id "
        "e.g. PXX_ and an underscore and also contains -START-STOP- in integer "
        "values in the Kaldi convention (units of 10ms).",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-i,--in_asr",
        type=str,
        metavar="STR",
        dest="in_asr",
        help="Path to the GPU-GSS dir "
        "containing ./enhanced/*.flac enhanced files and "
        "a cuts_per_segment.json.gz manifest.",
    )
    parser.add_argument(
        "-o,--output_name",
        type=str,
        metavar="STR",
        dest="output_name",
        help="Path and filename for the JSON file with all "
        "the predictions (could contain multiple sessions), "
        "e.g. /tmp/chime6_gss_dev will create in /tmp "
        "/tmp/chime6_gss_dev.json.",
    )

    parser.add_argument(
        "-r,--sess_regex",
        type=str,
        default="(S[0-9]+)",
        metavar="STR",
        dest="sess_regex",
        help="Regular expression for parsing the session id inside the "
        "utterance id, default is (S[0-9]+) for CHiME-6 like sessions.",
    )

    args = parser.parse_args()
    parse2json(args.in_asr, args.output_name, args.sess_regex)
