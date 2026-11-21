#!/usr/bin/env python3
"""Two-speaker mixtures from mini_an4 utterances, for the diarization CI test.

Every pair of utterances from two different speakers of a split is mixed: the
first starts at 0, the second at a random offset that makes it overlap the
first, follow it after a pause, or (for the first pair of each split) start at
the same time. The result is a Kaldi diarization data directory:

    wav.scp   <rec> <path to the mixture wav>
    segments  <spk>-<rec> <rec> <start> <end>   one segment per speaker
    utt2spk   <spk>-<rec> <spk>
    spk2utt

With --vad, the data directory has one speaker, "speech", whose segments are
the union of the two utterances (one segment, or two when there is a pause):
the labels of a voice activity detector trained as diarization with
num_spk 1.

Usage:
    simulate_mixtures.py [--vad] <an4_wav_dir> <split_subdir> <wav_out_dir> <data_dir>
"""

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import soundfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("an4_wav_dir", type=Path, help="downloads/an4/wav")
    ap.add_argument("split", help="an4_clstk (train) or an4test_clstk (test)")
    ap.add_argument("wav_out_dir", type=Path)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--vad",
        action="store_true",
        help="write one speaker, speech, with the union of the utterances",
    )
    a = ap.parse_args()

    rng = random.Random(a.seed)
    utts = []  # (speaker, utterance id, waveform, fs)
    for path in sorted((a.an4_wav_dir / a.split).glob("*/*.sph")):
        wav, fs = soundfile.read(path, dtype="float32")
        utts.append((path.parent.name, path.stem, wav, fs))
    pairs = [(u, v) for u, v in itertools.combinations(utts, 2) if u[0] != v[0]]
    if not pairs:
        raise RuntimeError(f"no two speakers in {a.an4_wav_dir / a.split}")

    a.wav_out_dir.mkdir(parents=True, exist_ok=True)
    a.data_dir.mkdir(parents=True, exist_ok=True)
    wav_scp, segments, utt2spk = [], [], []
    for i, (u, v) in enumerate(pairs):
        (spk1, id1, x1, fs), (spk2, id2, x2, fs2) = u, v
        assert fs == fs2, (fs, fs2)
        if i == 0:
            offset = 0  # both start together
        else:
            # overlap the first utterance, or follow it after up to 0.5 s
            offset = rng.randint(len(x1) // 4, len(x1) + fs // 2)
        mix = np.zeros(max(len(x1), offset + len(x2)), dtype=np.float32)
        mix[: len(x1)] += x1
        mix[offset : offset + len(x2)] += x2
        peak = np.abs(mix).max()
        if peak > 0.9:
            mix *= 0.9 / peak
        rec = f"mix{i:02d}_{id1}_{id2}"
        out = a.wav_out_dir / f"{rec}.wav"
        soundfile.write(out, mix, fs, subtype="PCM_16")
        wav_scp.append(f"{rec} {out}")
        if a.vad:
            # the second utterance starts at or after the first one
            if offset <= len(x1):
                spans = [(0, max(len(x1), offset + len(x2)))]
            else:
                spans = [(0, len(x1)), (offset, offset + len(x2))]
            labels = [
                (f"speech-{rec}-{k}", "speech", s, e) for k, (s, e) in enumerate(spans)
            ]
        else:
            labels = [
                (f"{spk1}-{rec}", spk1, 0, len(x1)),
                (f"{spk2}-{rec}", spk2, offset, offset + len(x2)),
            ]
        for seg, spk, start, end in labels:
            # Write the times with 7 decimals: at 16 kHz a sample is 0.0000625 s,
            # so every sample boundary is written exactly.
            segments.append(f"{seg} {rec} {start / fs:.7f} {end / fs:.7f}")
            utt2spk.append(f"{seg} {spk}")

    (a.data_dir / "wav.scp").write_text("\n".join(sorted(wav_scp)) + "\n")
    (a.data_dir / "segments").write_text("\n".join(sorted(segments)) + "\n")
    (a.data_dir / "utt2spk").write_text("\n".join(sorted(utt2spk)) + "\n")
    spk2utt = {}
    for line in sorted(utt2spk):
        seg, spk = line.split()
        spk2utt.setdefault(spk, []).append(seg)
    (a.data_dir / "spk2utt").write_text(
        "".join(f"{spk} {' '.join(segs)}\n" for spk, segs in sorted(spk2utt.items()))
    )
    print(f"{a.data_dir}: {len(pairs)} mixtures")


if __name__ == "__main__":
    main()
