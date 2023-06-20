import argparse
import re
from pathlib import Path

from soundfile import SoundFile

parser = argparse.ArgumentParser()
parser.add_argument("wav_scp", type=str)
parser.add_argument("noise_dir", type=str)
parser.add_argument("annotation_dir", type=str)
parser.add_argument("--sample_rate", default="16k", type=str, choices=["16k", "48k"])
parser.add_argument("--outfile", type=str)
args = parser.parse_args()


fs = 16000 if args.sample_rate == "16k" else 48000
wav_scp = {}
with open(args.wav_scp, "r") as f:
    for line in f:
        uid, path = line.strip().split(maxsplit=1)
        wav_scp[uid] = path

annotation = {}
for lst in Path(args.annotation_dir).glob(f"*dB_embeddedAnnotations{fs}.lst"):
    with lst.open("r") as f:
        for line in f:
            uid, noise, start, length = line.strip().split()
            noise_path = Path(args.noise_dir) / f"{noise}.wav"
            assert noise_path.exists(), noise_path
            annotation[uid] = (str(noise_path), int(start), int(length))

with open(args.outfile, "w") as f:
    for uid, wav in wav_scp.items():
        match = re.fullmatch(r"^s(\d+)(_\w+)_m?\d+dB$", uid)
        if match:
            uid_org = f"s{int(match.group(1))}{match.group(2)}"
        else:
            raise ValueError(f"{uid, wav, args.wav_scp, args.annotation_dir}")
        noise, start, length = annotation[uid_org]

        with SoundFile(noise) as f:
            dur_noise = f.frames
        assert start + length <= dur_noise, (start, length, dur_noise)

        with SoundFile(wav) as f:
            dur_wav = f.frames
        assert dur_wav == length, (dur_wav, length)

        f.write(f"{uid} sox {noise} -t wav - trim {start} {length}s |\n")
