#!/usr/bin/env python

import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ForceAlignmentInfo(object):
    tokens: List[str]
    frame_durations: List[int]
    start_sec: Optional[float]
    end_sec: Optional[float]


def get_mfa_alignment_by_sample_id(
    textgrid_zip_path: str,
    sample_id: str,
    sample_rate: int,
    hop_length: int,
    silence_phones: List[str] = ("sil", "sp", "spn"),
) -> ForceAlignmentInfo:
    try:
        import tgt
    except ImportError:
        raise ImportError("Please install TextGridTools: pip install tgt")

    filename = f"{sample_id}.TextGrid"
    out_root = Path(tempfile.gettempdir())
    tgt_path = out_root / filename
    with zipfile.ZipFile(textgrid_zip_path) as f_zip:
        f_zip.extract(filename, path=out_root)
    textgrid = tgt.io.read_textgrid(tgt_path.as_posix())
    os.remove(tgt_path)

    phones, frame_durations = [], []
    start_sec, end_sec, end_idx = 0, 0, 0
    for t in textgrid.get_tier_by_name("phones")._objects:
        s, e, p = t.start_time, t.end_time, t.text
        # Trim leading silences
        if len(phones) == 0:
            if p in silence_phones:
                continue
            else:
                start_sec = s
        phones.append(p)
        if p not in silence_phones:
            end_sec = e
            end_idx = len(phones)
        r = sample_rate / hop_length
        frame_durations.append(int(np.round(e * r) - np.round(s * r)))
    # Trim tailing silences
    phones = phones[:end_idx]
    frame_durations = frame_durations[:end_idx]

    return ForceAlignmentInfo(
        tokens=phones,
        frame_durations=frame_durations,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def main():
    try:
        import tgt
    except ImportError:
        raise ImportError("Please install TextGridTools: pip install tgt")
    args = sys.argv
    silence_phones: List[str] = ("sil", "sp", "spn", "")
    alignment_path = args[1]
    wav_scp_path = args[2]

    output_wav_scp_path = f"{wav_scp_path}.out"

    wav_scp = open(wav_scp_path, "r")
    output_wav_scp = open(output_wav_scp_path, "w")
    for line in wav_scp.readlines():
        id, path = line.strip().split(maxsplit=1)
        filename = f"{id}.TextGrid"
        tgt_path = f"{alignment_path}/{filename}"
        if not os.path.exists(path):
            print(f"{path} does not exist. Omitting...")
            continue
        try:
            textgrid = tgt.io.read_textgrid(tgt_path)
        except FileNotFoundError:
            print(f"{tgt_path} does not have an alignment! Omitting...")
            continue
        firstPhoneDone = False
        start_sec, end_sec = 0, 0
        for t in textgrid.get_tier_by_name("phones")._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # Trim leading silences
            if not firstPhoneDone:
                if p in silence_phones:
                    continue
                else:
                    start_sec = s
            firstPhoneDone = True
            if p not in silence_phones:
                end_sec = e
        sox_cmd = f"sox {path} -t wav - trim {start_sec} {end_sec-start_sec} | "
        output_line = " ".join([id, sox_cmd]) + "\n"
        output_wav_scp.write(output_line)
    os.rename(output_wav_scp_path, wav_scp_path)


if __name__ == "__main__":
    main()
