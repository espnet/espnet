# convert_trial.py

# Makes ESPnet trial files from ASVspoof5 protocol
# output files: trial.scp, trial2.scp, trial3.scp, trial4.scp, trial_label
# enrollment utterance paths are in trial.scp, trial2.scp, trial3.scp
# if less than 3 enrollment utterances, the last path is repeated
# test utterance paths are in trial4.scp

# label mapping used:
# bonafide: 0
# nontarget: 1
# spoof: 2

import argparse
import os
import sys

# label mapping dictionary
label_dict = {}
label_dict["target"] = 0
label_dict["nontarget"] = 1
label_dict["spoof"] = 2

def main(args):
    with open(args.trial, "r") as f:
        lines_trial_org = f.readlines()
    with open(args.scp, "r") as f:
        lines_scp = f.readlines()
    with open(args.enroll, "r") as f:
        lines_enroll = f.readlines()

    scp_dict = dict()
    for scp in lines_scp:
        utt_id, path = scp.strip().split(" ")
        # if utt_id has a dash, take the part after the dash
        if "-" in utt_id:
            utt_id = utt_id.split("-")[1]
        scp_dict[utt_id] = path
    
    enroll_dict = dict()
    for enroll in lines_enroll:
        # enrollment file is of the form: speakerID utt1,utt2,utt3
        # in some cases there are less than 3 enrollment utterances
        # if less than 3, the last given enrollment utterance is repeated
        speakerID, enroll_utt = enroll.strip().split()
        enroll_utt = enroll_utt.split(",")
        if len(enroll_utt) < 3:
            enroll_utt.extend([enroll_utt[-1]]*(3-len(enroll_utt)))
        enroll_dict[speakerID] = enroll_utt

    with open(os.path.join(args.out, "trial.scp"), "w") as f_trial, open(
        os.path.join(args.out, "trial2.scp"), "w") as f_trial2, open(
            os.path.join(args.out, "trial3.scp"), "w") as f_trial3, open(
                os.path.join(args.out, "trial4.scp"), "w") as f_trial4, open(
                    os.path.join(args.out, "trial_label"), "w") as f_label:
        for tr in lines_trial_org:
            if args.task == "dev":
                enrolled_speaker, test_utt, label = tr.strip().split()
            else:
                enrolled_speaker, test_utt = tr.strip().split()
            # each trial is identified by a joint key of form: speakerID*test_utt
            key = f"{enrolled_speaker}*{test_utt}"
            # write trial.scp, trial2.scp, trial3.scp
            f_trial.write(f"{key} {scp_dict[enroll_dict[enrolled_speaker][0]]}\n")
            f_trial2.write(f"{key} {scp_dict[enroll_dict[enrolled_speaker][1]]}\n")
            f_trial3.write(f"{key} {scp_dict[enroll_dict[enrolled_speaker][2]]}\n")
            # write trial4.scp
            f_trial4.write(f"{key} {scp_dict[test_utt]}\n")
            # write trial_label for dev task
            if args.task == "dev":
                f_label.write(f"{key} {label_dict[label]}\n")
            # write dummy labels for eval task
            if args.task == "eval": 
                f_label.write(f"{key} 0\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial mapper")
    parser.add_argument(
        "--trial",
        type=str,
        required=True,
        help="path to the trial file",
    )
    parser.add_argument(
        "--enroll",
        type=str,
        required=True,
        help="path to the enrollment file",
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
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="task: dev or eval",
    )
    args = parser.parse_args()

    sys.exit(main(args))