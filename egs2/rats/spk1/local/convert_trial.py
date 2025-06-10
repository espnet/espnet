import argparse
import os
import pickle as pk
import sys


def main(args):
    with open(args.trial, "r") as f:
        lines_trial_org = f.readlines()
    with open(args.scp, "r") as f:
        lines_scp = f.readlines()

    src2spk = pk.load(open(args.src2spk, "rb"))
    d_src2spk = {key: src2spk[key] for key in src2spk}

    scp_dict = dict()
    for scp in lines_scp:
        utt_id, path = scp.strip().split(" ")
        f_name = utt_id.split("/")[-1]
        spk = d_src2spk[f_name.split("_")[0]]
        scp_dict[utt_id] = path
    trial_set = set()

    with open(os.path.join(args.out, "trial.scp"), "w") as f_trial, open(
        os.path.join(args.out, "trial2.scp"), "w"
    ) as f_trial2, open(os.path.join(args.out, "trial_label"), "w") as f_label:
        for tr in lines_trial_org:
            label, utt1, utt2 = tr.strip().split(" ")
            utt1_id = "/".join(utt1.split("/")[-3:])[:-5]
            f_utt1 = utt1_id.split("/")[-1]
            utt1_spk = d_src2spk[f_utt1.split("_")[0]]
            utt1 = utt1_spk + "/" + utt1_id

            utt2_id = "/".join(utt2.split("/")[-3:])[:-5]
            f_utt2 = utt2_id.split("/")[-1]
            utt2_spk = d_src2spk[f_utt2.split("_")[0]]
            utt2 = utt2_spk + "/" + utt2_id

            joint_key = "*".join([utt1, utt2])
            if joint_key in trial_set:
                break
            trial_set.add(joint_key)
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
        "--src2spk",
        type=str,
        required=True,
        help="directory of src2spk file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="destinatino directory of processed trial and label files",
    )
    args = parser.parse_args()

    sys.exit(main(args))
