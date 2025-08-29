# filter babel dev data, remove the data lower than 10 seconds
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--babel_dir", type=str, required=True)
    parser.add_argument("--babel_over_10s_dir", type=str, required=True)
    return parser.parse_args()


args = parse_args()

babel_dir = args.babel_dir
babel_over_10s_dir = args.babel_over_10s_dir
fs = 16000
lower_bound = 10 * fs
os.makedirs(babel_over_10s_dir, exist_ok=True)

utt2num_samples_dir = os.path.join(babel_dir, "utt2num_samples")
utt_over_10s = set()
with open(utt2num_samples_dir, "r") as f:
    for line in f:
        utt_id, num_samples = line.strip().split()
        if int(num_samples) >= lower_bound:
            utt_over_10s.add(utt_id)

kaldi_files = ["wav.scp", "utt2spk", "utt2num_samples"]

for kaldi_file in kaldi_files:
    kaldi_file_path = os.path.join(babel_dir, kaldi_file)
    kaldi_file_over_10s_path = os.path.join(babel_over_10s_dir, kaldi_file)
    filtered_lines = []
    with open(kaldi_file_path, "r") as f:
        for line in f:
            utt_id = line.strip().split()[0]
            if utt_id in utt_over_10s:
                filtered_lines.append(line)
    with open(kaldi_file_over_10s_path, "w") as f:
        f.writelines(sorted(filtered_lines))
