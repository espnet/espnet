# convert_trial.py

# Makes ESPnet trial files from ASVspoof5 protocol

import argparse
import os
import sys


def main(args):
    with open(args.trial, "r") as f:
        lines_trial_org = f.readlines()
    with open(args.scp, "r") as f:
        lines_scp = f.readlines()

    scp_dict = dict()
    for scp in lines_scp:
        utt_id, path = scp.strip().split(" ")
        # if utt_id has a dash, take the part after the dash
        if "-" in utt_id:
            utt_id = utt_id.split("-")[1]
        scp_dict[utt_id] = path

    with open(os.path.join(args.out, "trial.scp"), "w") as f_trial, open(
        os.path.join(args.out, "trial2.scp"), "w"
    ) as f_trial2, open(os.path.join(args.out, "trial_label"), "w") as f_label:
        for tr in lines_trial_org:
            enrolment, test_utt, label = tr.strip().split(" ")
            joint_key = "*".join([enrolment, test_utt])
            f_trial.write(f"{joint_key} {scp_dict[enrolment]}\n")
            f_trial2.write(f"{joint_key} {scp_dict[test_utt]}\n")
            f_label.write(f"{joint_key} {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial mapper")
    parser.add_argument(
        "--trial",
        type=str,
        required=True,
        help="directory of the original trial file",
    )
    parser.add_argument(
        "--scp",
        type=str,
        required=True,
        help="directory of wav.scp file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="destination directory of processed trial and label files",
    )
    args = parser.parse_args()

    sys.exit(main(args))