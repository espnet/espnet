import argparse
import glob
import json
import os.path
import warnings
from pathlib import Path


def rttm2json(rttm_file):
    with open(rttm_file, "r") as f:
        rttm = f.readlines()

    rttm = [x.rstrip("\n") for x in rttm]
    filename = Path(rttm_file).stem

    to_json = []
    for line in rttm:
        current = line.split(" ")
        start = current[3]
        duration = current[4]
        stop = str(float(start) + float(duration))
        speaker = current[7]
        session = filename
        to_json.append(
            {
                "session_id": session,
                "speaker": speaker,
                "start_time": start,
                "end_time": stop,
            }
        )

    to_json = sorted(to_json, key=lambda x: x["start_time"])
    return to_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r, --rttm_folder",
        type=str,
        default="/home/samco/dgx/CHiME6/decode_chime7/espnet/egs2/chime7_task1/asr1/exp/popcornell/chime7_task1_asr1_baseline/decode_asr_transformer_asr_model_valid.acc.ave/scoring/mixer6/20090714_134807_LDC_120290",
        metavar="STR",
        dest="rttm_folder",
        help="Path to folder containing .rttm files.",
    )
    parser.add_argument(
        "-o, --out_folder",
        type=str,
        default="",
        required=False,
        metavar="STR",
        dest="out_folder",
        help="Path to output folder. If not specified, same as input.",
    )

    args = parser.parse_args()

    output_folder = args.out_folder
    if not output_folder:
        output_folder = args.rttm_folder
    else:
        Path(output_folder).mkdir(exist_ok=True, parents=True)

    rttm_files = glob.glob(os.path.join(args.rttm_folder, "*.rttm"))
    if len(rttm_files) == 0:
        warnings.warn("No .rttm files found in {}".format(args.rttm_folder))

    for file in rttm_files:
        to_json = rttm2json(file)
        filename = Path(file).stem
        with open(os.path.join(output_folder, filename + ".json"), "w") as f:
            json.dump(to_json, f, indent=4)
