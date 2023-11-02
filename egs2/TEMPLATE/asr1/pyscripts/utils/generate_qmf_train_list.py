import sys

import numpy as np
import soundfile as sf
import yaml

np.random.seed(0)


def load_yaml(yamlfile):
    with open(yamlfile, "r") as stream:
        data = yaml.safe_load(stream)
        return data


def main(args):
    spk2utt = args[0]
    wav_scp = args[1]
    out_dir = args[2]
    cfg = load_yaml(args[3])
    utt2spk = args[4]
    cohort_list = args[5]
    samp_rate = args[6][:-1]
    print(cfg)
    with open(wav_scp) as f:
        lines = f.readlines()
    wav2dir_dic = {
        line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in lines
    }

    with open(spk2utt, "r") as f:
        spk2utt = f.readlines()[: cfg["num_cohort_spk"]]

    out_utts = set()
    trg_samp = int(cfg["target_duration"] * int(samp_rate) * 1000)

    # get list of utterances used on cohort set to remove them from qmf trainset
    with open(cohort_list) as f:
        cohort_list = f.readlines()
    cohort_utts = set()
    for line in cohort_list:
        utt1, utt2 = line.strip().split(" ")[0].split("*")
        cohort_utts.add(utt1)
        cohort_utts.add(utt2)

    spk2utt_short, spk2utt_long = {}, {}
    spk_list_whole = []
    for spk in spk2utt:
        chunk = spk.strip().split(" ")
        spk = chunk[0]
        spk_list_whole.append(spk)
        spk2utt_short[spk] = []
        spk2utt_long[spk] = []
        utts = sorted(chunk[1:])
        for utt in utts:
            # exclude if utt was already used for the cohort set
            if utt in cohort_utts:
                continue

            utt_file = wav2dir_dic[utt]
            dur = sf.info(utt_file).duration
            if dur < 2:
                continue

            if dur >= cfg["qmf_dur_thresh"]:
                spk2utt_long[spk].append(utt)
            else:
                spk2utt_short[spk].append(utt)

    # filter out empty speakers
    for spk in spk_list_whole:
        if len(spk2utt_short[spk]) < 2:
            del spk2utt_short[spk]
    for spk in spk_list_whole:
        if len(spk2utt_long[spk]) < 2:
            del spk2utt_long[spk]
    spk_list_short = list(spk2utt_short.keys())
    spk_list_long = list(spk2utt_long.keys())

    with open(out_dir + "/qmf_train.scp", "w") as f_qmf, open(
        out_dir + "/qmf_train2.scp", "w"
    ) as f_qmf2, open(out_dir + "/qmf_train_speech_shape", "w") as f_shape, open(
        out_dir + "/qmf_train_label", "w"
    ) as f_lbl:
        # replace = True if len(spk_list) >= cfg["qmf_n_trial_per_condition"] else False
        spk2utt_short_used = {}
        spk2utt_long_used = {}
        for spk in spk_list_whole:
            spk2utt_short_used[spk] = set()
            spk2utt_long_used[spk] = set()

        # generate short-short target trials
        sel_spks = np.random.choice(
            spk_list_short, cfg["qmf_num_trial_per_condition"], replace=True
        )
        for spk in sel_spks:
            utt1, utt2 = np.random.choice(spk2utt_short[spk], 2, replace=False)
            spk2utt_short_used[spk].add(utt1)
            spk2utt_short_used[spk].add(utt2)
            if f"{utt1}*{utt2}" in out_utts:
                continue
            else:
                out_utts.add(f"{utt1}*{utt2}")
            f_qmf.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_qmf2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 1\n")

        # generate long-long target trials
        sel_spks = np.random.choice(
            spk_list_long, cfg["qmf_num_trial_per_condition"], replace=True
        )
        for spk in sel_spks:
            utt1, utt2 = np.random.choice(spk2utt_long[spk], 2, replace=False)
            spk2utt_long_used[spk].add(utt1)
            spk2utt_long_used[spk].add(utt2)
            if f"{utt1}*{utt2}" in out_utts:
                continue
            else:
                out_utts.add(f"{utt1}*{utt2}")
            f_qmf.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_qmf2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 1\n")

        sel_spks = np.random.choice(
            spk_list_whole, cfg["qmf_num_trial_per_condition"], replace=True
        )
        # generate short-long target trials
        for spk in sel_spks:
            if spk not in spk_list_short or spk not in spk_list_long:
                continue
            utt1 = np.random.choice(spk2utt_long[spk], 1)[0]
            utt2 = np.random.choice(spk2utt_short[spk], 1)[0]
            spk2utt_long_used[spk].add(utt1)
            spk2utt_short_used[spk].add(utt2)
            if f"{utt1}*{utt2}" in out_utts:
                continue
            else:
                out_utts.add(f"{utt1}*{utt2}")
            f_qmf.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_qmf2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 1\n")

        # filter out empty speakers
        for spk in spk_list_whole:
            if len(spk2utt_short_used[spk]) < 2:
                del spk2utt_short_used[spk]
                continue
            spk2utt_short_used[spk] = list(spk2utt_short_used[spk])
        for spk in spk_list_whole:
            if len(spk2utt_long_used[spk]) < 2:
                del spk2utt_long_used[spk]
                continue
            spk2utt_long_used[spk] = list(spk2utt_long_used[spk])
        spk_list_short_used = list(spk2utt_short_used.keys())
        spk_list_long_used = list(spk2utt_long_used.keys())

        # generate short-short non-target trials
        for i in range(cfg["qmf_num_trial_per_condition"]):
            spk1, spk2 = np.random.choice(spk_list_short_used, 2, replace=False)
            utt1 = np.random.choice(spk2utt_short_used[spk1], 1)[0]
            utt2 = np.random.choice(spk2utt_short_used[spk2], 1)[0]
            if f"{utt1}*{utt2}" in out_utts:
                continue
            else:
                out_utts.add(f"{utt1}*{utt2}")
            f_qmf.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_qmf2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 0\n")

        # generate long-long non-target trials
        for i in range(cfg["qmf_num_trial_per_condition"]):
            spk1, spk2 = np.random.choice(spk_list_long_used, 2, replace=False)
            utt1 = np.random.choice(spk2utt_long_used[spk1], 1)[0]
            utt2 = np.random.choice(spk2utt_long_used[spk2], 1)[0]
            if f"{utt1}*{utt2}" in out_utts:
                continue
            else:
                out_utts.add(f"{utt1}*{utt2}")
            f_qmf.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_qmf2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 0\n")

        # generate short-long non-target trials
        for i in range(cfg["qmf_num_trial_per_condition"]):
            spk1 = np.random.choice(spk_list_short_used, 1, replace=False)[0]
            spk2 = np.random.choice(spk_list_long_used, 1, replace=False)[0]
            if spk1 == spk2:
                continue
            utt1 = np.random.choice(spk2utt_short_used[spk1], 1)[0]
            utt2 = np.random.choice(spk2utt_long_used[spk2], 1)[0]
            if f"{utt1}*{utt2}" in out_utts:
                continue
            else:
                out_utts.add(f"{utt1}*{utt2}")
            f_qmf.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_qmf2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 0\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
