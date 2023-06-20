import argparse
import re

from soundfile import SoundFile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("cln_wav_scp", type=str)
parser.add_argument("noisy_wav_scp", type=str)
parser.add_argument("--audio_format", default=".wav", type=str)
parser.add_argument("--outfile", type=str)
args = parser.parse_args()

with open(args.cln_wav_scp, "r") as f:
    cln_wav_scp = [line.rstrip() for line in f.readlines()]
with open(args.noisy_wav_scp, "r") as f:
    noisy_wav_scp = [line.rstrip() for line in f.readlines()]
assert len(noisy_wav_scp) == len(cln_wav_scp)

with open(args.outfile, "w") as f:
    for cln_scp, noisy_scp in tqdm(zip(cln_wav_scp, noisy_wav_scp)):
        uid1, cln = cln_scp.split(maxsplit=1)
        uid2, noisy = noisy_scp.split(maxsplit=1)
        assert uid1 == uid2, (uid1, uid2)

        pattern = r"^\s*sox\s."
        assert not re.match(pattern, cln), cln
        if re.match(pattern, noisy):
            for n in noisy.split():
                if n.endswith(args.audio_format):
                    noisy = n
                    break
            else:
                raise ValueError(noisy)

        with SoundFile(cln) as f1, SoundFile(noisy) as f2:
            dur_cln, dur_noisy = f1.frames, f2.frames
        assert dur_cln <= dur_noisy, (dur_cln, dur_noisy)

        if dur_cln == dur_noisy:
            f.write(f"{uid1} {cln}\n")
        elif dur_cln < dur_noisy:
            f.write(f"{uid1} sox {cln} -t wav - pad 0 {dur_noisy - dur_cln}s |\n")
        else:
            f.write(f"{uid1} sox {cln} -t wav - trim 0 {dur_noisy}s |\n")
