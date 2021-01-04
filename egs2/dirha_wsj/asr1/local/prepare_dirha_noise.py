#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import resampy
import soundfile


def prepare(
    dirha_dir: str,
    minimum_duration: float,
    fs: int,
    audio_dir: str,
    data_dir: Optional[str],
    num_files_per_dir: int = 10,
    audio_format: str = "flac",
):
    dirha_dir = Path(dirha_dir)
    audio_dir = Path(audio_dir)
    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        fscp = (data_dir / "wav.scp").open("w")
    else:
        fscp = None

    joint = None
    num_files = 0
    num_dir = 0
    for w in dirha_dir.glob("**/*.wav"):
        x, r = soundfile.read(w)

        # Concatenate noises until
        # it has 20seconds durations at minimum
        if joint is None:
            joint = x
        else:
            joint = np.concatenate([joint, x])

        if (joint.shape[0] / float(r)) >= minimum_duration:
            if r != fs:
                joint = resampy.resample(joint, r, fs, axis=0)
            num_files += 1
            if num_files % num_files_per_dir == 0:
                num_dir += 1
            owav = audio_dir / f"dir.{num_dir}" / f"noise_{num_files}.{audio_format}"
            owav.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(owav, joint, fs)

            if fscp is not None:
                fscp.write(f"noise_{num_files} {owav}\n")
            joint = None

    else:
        if joint is not None and (joint.shape[0] / float(r)) < minimum_duration:
            if r != fs:
                joint = resampy.resample(joint, r, fs, axis=0)
            # Read the last saved file
            owav = audio_dir / f"dir.{num_dir}" / f"noise_{num_files}.{audio_format}"
            x, r = soundfile.read(owav)
            joint = np.concatenate([x, joint])
            soundfile.write(owav, joint, fs)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Dirha WSJ data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dirha_dir", required=True, help="Input directory")
    parser.add_argument("--audio_dir", required=True, help="Output directory")
    parser.add_argument("--data_dir", help="Output directory")
    parser.add_argument("--minimum_duration", type=float, default=20)
    parser.add_argument("--audio_format", default="flac")
    parser.add_argument("--fs", type=int, default=16000)
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    prepare(**kwargs)


if __name__ == "__main__":
    main()
