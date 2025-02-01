# make_feats_scp.py

# This script reads the input directory and creates the feats.scp file which is of form:
# <utt_id> <absolute_path_to_npy_file>


import argparse
import os

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--trial", type=str, required=True, help="path to the trial file"
    )
    parser.add_argument(
        "--enroll", type=str, required=True, help="path to the enrollment file"
    )
    parser.add_argument("--task", type=str, required=True, help="Task: dev or eval")
    return parser.parse_args()


def main():
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    enroll_file = args.enroll
    trial_file = args.trial

    # generate feats.scp file
    with open(os.path.join(output_dir, "feats.scp"), "w") as f_feats:
        for root, _, files in os.walk(input_dir):
            for file in tqdm(files):
                if file.endswith(".npy"):
                    utt_id = file.split(".")[0]
                    f_feats.write(
                        f"{utt_id} {os.path.abspath(os.path.join(root, file))}\n"
                    )

    print("feats.scp file created successfully!")

    # read the enroll file
    with open(enroll_file, "r") as f:
        lines_enroll = f.readlines()

    # read the trial file
    with open(trial_file, "r") as f:
        lines_trial_org = f.readlines()

    # create a dictionary of feats.scp file
    scp_dict = {}
    with open(os.path.join(output_dir, "feats.scp"), "r") as f:
        for line in f:
            utt_id, path = line.strip().split(" ")
            # if utt_id has a dash, take the part after the dash
            if "-" in utt_id:
                utt_id = utt_id.split("-")[1]
            scp_dict[utt_id] = path

    # create a dictionary of enroll file
    enroll_dict = {}
    # enrollment file is of the form: speakerID utt1,utt2,utt3
    # in some cases there are less than 3 enrollment utterances
    # if less than 3, the last given enrollment utterance is repeated
    for enroll in lines_enroll:
        speakerID, enroll_utt = enroll.strip().split()
        enroll_utt = enroll_utt.split(",")
        if len(enroll_utt) < 3:
            enroll_utt.extend([enroll_utt[-1]] * (3 - len(enroll_utt)))
        enroll_dict[speakerID] = enroll_utt

    # make feats1.scp, feats2.scp, feats3.scp, feats4.scp
    with open(os.path.join(output_dir, "feats1.scp"), "w") as f_feats1, open(
        os.path.join(output_dir, "feats2.scp"), "w"
    ) as f_feats2, open(os.path.join(output_dir, "feats3.scp"), "w") as f_feats3, open(
        os.path.join(output_dir, "feats4.scp"), "w"
    ) as f_feats4:
        for tr in lines_trial_org:
            if args.task == "dev":
                enrolled_speaker, test_utt, label = tr.strip().split()
            else:
                enrolled_speaker, test_utt = tr.strip().split()
            # each trial is identified by a joint key of form: speakerID*test_utt
            key = f"{enrolled_speaker}*{test_utt}"
            # write feats1.scp, feats2.scp, feats3.scp
            f_feats1.write(f"{key} {scp_dict[enroll_dict[enrolled_speaker][0]]}\n")
            f_feats2.write(f"{key} {scp_dict[enroll_dict[enrolled_speaker][1]]}\n")
            f_feats3.write(f"{key} {scp_dict[enroll_dict[enrolled_speaker][2]]}\n")
            # write feats4.scp
            f_feats4.write(f"{key} {scp_dict[test_utt]}\n")


if __name__ == "__main__":
    main()
