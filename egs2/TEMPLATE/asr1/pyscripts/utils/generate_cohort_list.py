import sys
import yaml
import soundfile as sf

def load_yaml(yamlfile):
    with open(yamlfile, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)

def main(args):
    spk2utt = args[0]
    wav_scp = args[1]
    out_dir = args[2]
    cfg = load_yaml(args[3])
    wav2dir_dic = {line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in lines}

    with open(spk2utt, "r") as f:
        spk2utt = spk2utt.readlines()

    spk2utt_dic = {} # with num_utt_per_spk utts
    for spk in spk2utt:
        chunk = spk.strip().split(" ")
        spk = chunk[0]
        spk2utt_dic[spk] = []
        utts = sorted(chunk[1:])
        for utt in utts:
            utt_file = wav2dir_dic[utt]
            dur = sf.info(utt_file).duration
            if dur >= cfg["utt_select_sec"]:
                spk2utt_dic[spk].append(utt)




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

