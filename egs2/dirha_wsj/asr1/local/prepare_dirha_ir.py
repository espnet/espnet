#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import resampy
import scipy.io
import soundfile


def prepare(
    dirha_dir: str,
    fs: int,
    audio_dir: str,
    data_dir: Optional[str],
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

    for m in dirha_dir.glob("**/*.mat"):
        # FIXME(kamo): Using all IRs here regardless of mic type
        if m.stem == "ref-chirp":
            continue
        # (Time, Channel) or (Channel, Time)
        x = scipy.io.loadmat(m)["risp_imp"]
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            raise RuntimeError(m, x.shape)
        # 48khz impulse response
        r = 48000
        if r != fs:
            x = resampy.resample(x, r, fs, axis=0)
        # Rescale
        x = 0.95 * x / max(abs(x))
        owav = audio_dir / m.parent.name / f"{m.stem}.{audio_format}"
        owav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(owav, x, fs)
        if fscp is not None:
            rid = f"{m.parent.name}_{m.stem}"
            fscp.write(f"{rid} {owav}\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Dirha WSJ data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dirha_dir", required=True, help="Input directory")
    parser.add_argument("--audio_dir", required=True, help="Output directory")
    parser.add_argument("--data_dir", help="Output directory")
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
