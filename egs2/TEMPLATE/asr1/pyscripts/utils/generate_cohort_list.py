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
    samp_rate = args[4][:-1]
    print(cfg)
    with open(wav_scp) as f:
        lines = f.readlines()
    wav2dir_dic = {
        line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in lines
    }

    with open(spk2utt, "r") as f:
        spk2utt = f.readlines()[: cfg["num_cohort_spk"]]

    utt_list = []
    trg_samp = int(cfg["target_duration"] * int(samp_rate) * 1000)

    for spk in spk2utt:
        chunk = spk.strip().split(" ")
        spk = chunk[0]
        utts = sorted(chunk[1:])
        np.random.shuffle(utts)
        n_selected = 0
        for utt in utts:
            utt_file = wav2dir_dic[utt]
            dur = sf.info(utt_file).duration
            if dur >= cfg["utt_select_sec"]:
                utt_list.append(utt)
                n_selected += 1
                if n_selected == cfg["num_utt_per_spk"]:
                    break
    print(f"Cohort utterances selected, {len(utt_list)} utts, {len(spk2utt)} spks")

    # generate output adequate to ESPnet-SPK inference template
    utt_list1 = utt_list[: len(utt_list) // 2]
    utt_list2 = utt_list[len(utt_list) // 2 :]
    with open(out_dir + "/cohort.scp", "w") as f_coh, open(
        out_dir + "/cohort2.scp", "w"
    ) as f_coh2, open(out_dir + "/cohort_speech_shape", "w") as f_shape, open(
        out_dir + "/cohort_label", "w"
    ) as f_lbl:
        for utt1, utt2 in zip(utt_list1, utt_list2):
            f_coh.write(f"{utt1}*{utt2} {wav2dir_dic[utt1]}\n")
            f_coh2.write(f"{utt1}*{utt2} {wav2dir_dic[utt2]}\n")
            f_shape.write(f"{utt1}*{utt2} {trg_samp}\n")
            f_lbl.write(f"{utt1}*{utt2} 0\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
