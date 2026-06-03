import argparse
import os
import random
from collections import defaultdict


def load_info(info_path):
    scp = {}
    utt2spk = {}
    spk2utt = defaultdict(list)
    with open(os.path.join(info_path, "wav.scp"), "r") as f:
        for line in f:
            utt, path = line.strip().split(None, 1)
            scp[utt] = path
    with open(os.path.join(info_path, "utt2spk"), "r") as f:
        for line in f:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)
    return scp, utt2spk, spk2utt


def main(args):
    random.seed(args.seed)

    enroll_scp, enroll_utt2spk, enroll_spk2utt = load_info(args.enroll_dir)
    test_scp, utt2spk, spk2utt = load_info(args.test_dir)

    trials_per_enroll = args.trials_per_enroll
    target_ratio = args.target_ratio

    used_utts = set()
    trial_lines = []

    for enroll_utt in enroll_scp:
        spk = enroll_utt2spk[enroll_utt]
        target_pool = [utt for utt in spk2utt[spk] if utt in test_scp]
        n_target = int(trials_per_enroll * target_ratio)
        n_nontarget = trials_per_enroll - n_target

        # Target trials
        selected_targets = random.sample(target_pool, min(n_target, len(target_pool)))

        # Non-target trials
        other_utts = [utt for utt in test_scp if utt2spk[utt] != spk]
        selected_nontargets = random.sample(
            other_utts, min(n_nontarget, len(other_utts))
        )

        for utt in selected_targets:
            joint_key = f"{enroll_utt}*{utt}"
            trial_lines.append((joint_key, enroll_scp[enroll_utt], test_scp[utt], 1))
            used_utts.update([enroll_utt, utt])

        for utt in selected_nontargets:
            joint_key = f"{enroll_utt}*{utt}"
            trial_lines.append((joint_key, enroll_scp[enroll_utt], test_scp[utt], 0))
            used_utts.update([enroll_utt, utt])

    # Cover all test utterances
    all_test_utts = set(test_scp.keys())
    covered_test_utts = {utt for utt in used_utts if utt in test_scp}
    missing_test_utts = all_test_utts - covered_test_utts
    enroll_utts_list = list(enroll_scp.keys())

    for utt in sorted(missing_test_utts):
        n_trials = min(args.trials_per_missing, len(enroll_utts_list))
        enroll_utts = random.sample(enroll_utts_list, n_trials)
        for enroll_utt in enroll_utts:
            enroll_spk = enroll_utt2spk[enroll_utt]
            test_spk = utt2spk[utt]
            label = 0 if enroll_spk != test_spk else 1
            joint_key = f"{enroll_utt}*{utt}"
            trial_lines.append(
                (joint_key, enroll_scp[enroll_utt], test_scp[utt], label)
            )
            used_utts.update([enroll_utt, utt])

    print(f"[Info] {len(trial_lines)} trials, {len(used_utts)} unique utts")

    # Write trial files
    with (
        open(os.path.join(args.out_dir, "trial.scp"), "w") as f1,
        open(os.path.join(args.out_dir, "trial2.scp"), "w") as f2,
        open(os.path.join(args.out_dir, "trial_label"), "w") as flabel,
    ):
        for joint_key, path1, path2, label in trial_lines:
            f1.write(f"{joint_key} {path1}\n")
            f2.write(f"{joint_key} {path2}\n")
            flabel.write(f"{joint_key} {label}\n")

    # Write kaldi-style files
    utt2spk = {**enroll_utt2spk, **utt2spk}
    utt2spk_used = {utt: utt2spk[utt] for utt in used_utts}
    spk2utt_used = defaultdict(list)
    for utt, spk in utt2spk_used.items():
        spk2utt_used[spk].append(utt)

    with (
        open(os.path.join(args.out_dir, "wav.scp"), "w") as fwav,
        open(os.path.join(args.out_dir, "utt2spk"), "w") as futt2spk,
    ):
        for utt in sorted(used_utts):
            scp = enroll_scp.get(utt, test_scp.get(utt))
            fwav.write(f"{utt} {scp}\n")
            futt2spk.write(f"{utt} {utt2spk[utt]}\n")

    with open(os.path.join(args.out_dir, "spk2utt"), "w") as fspk2utt:
        for spk in sorted(spk2utt_used):
            fspk2utt.write(f"{spk} {' '.join(sorted(spk2utt_used[spk]))}\n")

    print(f"[Done] Trial generation complete. Output saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--trials_per_enroll", type=int, default=500)
    parser.add_argument("--trials_per_missing", type=int, default=3)
    parser.add_argument("--target_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
