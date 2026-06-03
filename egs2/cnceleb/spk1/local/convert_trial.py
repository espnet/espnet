import argparse
import os


def load_scp(scp_path):
    scp_dict = {}
    with open(scp_path, "r") as f:
        for line in f:
            utt_id, path = line.strip().split(None, 1)
            scp_dict[utt_id] = path
    return scp_dict


def main(args):
    """Convert trials.lst to trial.scp, trial2.scp, trial_label."""
    # Load wav.scp entries
    enroll_scp = load_scp(args.enroll_scp)
    test_scp = load_scp(args.test_scp)

    # Read trials.lst
    with open(args.trial, "r") as f:
        trials = [line.strip().split() for line in f if line.strip()]

    os.makedirs(args.out, exist_ok=True)

    with (
        open(os.path.join(args.out, "trial.scp"), "w") as f1,
        open(os.path.join(args.out, "trial2.scp"), "w") as f2,
        open(os.path.join(args.out, "trial_label"), "w") as flabel,
    ):

        for enroll_id, test_id, str_label in trials:
            test_id = test_id.split("/")[-1].split(".")[0]
            joint_key = f"{enroll_id}*{test_id}"
            label = 1 if str_label == "target" else 0

            if enroll_id not in enroll_scp:
                print(f"[Warning] Enroll '{enroll_id}' not found in enroll.scp")
                continue
            if test_id not in test_scp:
                print(f"[Warning] Test '{test_id}' not found in test.scp")
                continue

            f1.write(f"{joint_key} {enroll_scp[enroll_id]}\n")
            f2.write(f"{joint_key} {test_scp[test_id]}\n")
            flabel.write(f"{joint_key} {label}\n")

    print(f"[Done] Trial conversion complete. Files saved in: {args.out}")


def main_sep(args):
    """Convert trials.lst to trial.scp, trial2.scp, trial_label with enroll_map.

    This version handles multiple enrollments for a single enroll_id.
    enroll_map is expected to map enroll_id to a list of enrollments.
    The final trial.scp and trial2.scp will contain all combinations of
    enrollments and tests.
    """
    # Load wav.scp entries
    enroll_scp = load_scp(args.enroll_scp)
    test_scp = load_scp(args.test_scp)
    enroll_map = {}
    with open(args.enroll_map, "r") as f:
        for line in f:
            enrolls = line.strip().split()
            enroll_map[enrolls[0].split("-")[0]] = enrolls[1:]

    # Read trials.lst
    with open(args.trial, "r") as f:
        trials = [line.strip().split() for line in f if line.strip()]

    os.makedirs(args.out, exist_ok=True)

    with (
        open(os.path.join(args.out, "trial.scp"), "w") as f1,
        open(os.path.join(args.out, "trial2.scp"), "w") as f2,
        open(os.path.join(args.out, "trial_label"), "w") as flabel,
    ):

        for enroll_id, test_id, str_label in trials:
            enrolls = enroll_map[enroll_id.split("-")[0]]
            for enroll in enrolls:
                test_id = test_id.split("/")[-1].split(".")[0]
                joint_key = f"{enroll}*{test_id}"
                label = 1 if str_label == "target" else 0

                if enroll not in enroll_scp:
                    print(f"[Warning] Enroll '{enroll}' not found in enroll.scp")
                    continue
                if test_id not in test_scp:
                    print(f"[Warning] Test '{test_id}' not found in test.scp")
                    continue

                f1.write(f"{joint_key} {enroll_scp[enroll]}\n")
                f2.write(f"{joint_key} {test_scp[test_id]}\n")
                flabel.write(f"{joint_key} {label}\n")

    print(f"[Done] Trial conversion complete. Files saved in: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=str, required=True)
    parser.add_argument("--test_scp", type=str, required=True)
    parser.add_argument("--enroll_scp", type=str, required=True)
    parser.add_argument("--enroll_map", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if args.enroll_map is None:
        main(args)
    else:
        main_sep(args)
