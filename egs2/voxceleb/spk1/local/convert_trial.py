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
        scp_dict[utt_id] = path

    with open(os.path.join(args.out, "trial.scp"), "w") as f_trial, open(
        os.path.join(args.out, "trial2.scp"), "w"
    ) as f_trial2, open(os.path.join(args.out, "trial_label"), "w") as f_label:
        for tr in lines_trial_org:
            label, utt1, utt2 = tr.strip().split(" ")
            utt1 = utt1[:-4]
            utt2 = utt2[:-4]
            joint_key = "*".join([utt1, utt2])
            f_trial.write(f"{joint_key} {scp_dict[utt1]}\n")
            f_trial2.write(f"{joint_key} {scp_dict[utt2]}\n")
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
        help="destinatino directory of processed trial and label files",
    )
    args = parser.parse_args()

    sys.exit(main(args))
