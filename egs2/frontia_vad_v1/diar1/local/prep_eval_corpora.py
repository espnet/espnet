#!/usr/bin/env python3
"""VAD data directories of the evaluation-only corpora with published VAD results.

    python3 local/prep_eval_corpora.py CORPUS SPLIT OUT_DIR [paths ...]

Every recording is labelled with the union of all speakers' turns (one pseudo
speaker, as in the other corpora of this recipe). Multi-channel recordings are
evaluated on their first channel (local/select_channel.py in wav.scp).

    CORPUS        SPLIT      paths
    ami_mh        dev, eval  AMI_ROOT BUT_SETUP
        AMI Mix-Headset with the BUT diarization setup's only_words references
        (dev, test), the AMI condition of Bredin and Laurent, Interspeech 2021.
    alimeeting    dev, eval  ALIMEETING_DIR (holding Eval_Ali/ and Test_Ali/)
        far-field 8-channel audio (Eval = dev, Test = eval), TextGrid per meeting.
    aishell4      eval       AISHELL4_TEST (test/ with wav/ and TextGrid/*.rttm)
    voxconverse   dev, eval  WAV_ZIP_OR_DIR RTTM_DIR
        audio from the release zip (read with unzip -p) or a directory.
    earnings21    eval       EARNINGS21_ROOT (media/*.mp3, rttms/*.rttm; first channel)
    ava_speech    eval       AVA_SPEECH_ROOT (audio/<id>.flac holding 15:00-30:00 of
        each movie, ava_speech_labels_v1.csv); labels shift by -900 s and
        SPEECH_WITH_MUSIC / SPEECH_WITH_NOISE count as speech.
    ten_vad       eval       TESTSET_DIR (*.wav with *.scv frame labels)
"""

import argparse
import csv
import logging
import re
import shlex
import zipfile
from pathlib import Path

from vad_data_utils import (
    clip_intervals,
    merge_intervals,
    readable_audio_info,
    write_skipped,
    write_vad_dir,
)


def read_rttm(paths):
    turns = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            p = line.split()
            if p and p[0] == "SPEAKER":
                start = float(p[3])
                turns.setdefault(p[1], []).append((start, start + float(p[4])))
    return turns


def read_textgrid(path):
    """Non-empty intervals of every tier of a Praat TextGrid (long format)."""
    raw = Path(path).read_bytes()
    text = (
        raw.decode("utf-16")
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff")
        else raw.decode("utf-8")
    )
    out = []
    for tier in re.split(r"item \[\d+\]:", text)[1:]:
        for m in re.finditer(
            r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', tier, re.S
        ):
            if m.group(3).strip():
                out.append((float(m.group(1)), float(m.group(2))))
    return out


def channel_pipe(path, channel=0):
    return f"python3 local/select_channel.py {shlex.quote(str(path))} {channel} |"


def ami_mh(split, ami_root, setup):
    but = {"dev": "dev", "eval": "test"}[split]
    wav_root = ami_root / "wav_db" if (ami_root / "wav_db").is_dir() else ami_root
    turns = read_rttm(sorted(setup.glob(f"only_words/rttms/{but}/*.rttm")))
    files = {m: wav_root / m / "audio" / f"{m}.Mix-Headset.wav" for m in turns}
    return files, {f: str(p) for f, p in files.items()}, turns


def alimeeting(split, root):
    far = (
        root / {"dev": "Eval_Ali/Eval_Ali_far", "eval": "Test_Ali/Test_Ali_far"}[split]
    )
    files, entries, turns = {}, {}, {}
    for wav in sorted(far.glob("audio_dir/*.wav")):
        uid = wav.stem  # <room>_<meeting>_<mic>; TextGrid per <room>_<meeting>
        tg = far / "textgrid_dir" / ("_".join(uid.split("_")[:2]) + ".TextGrid")
        if tg.exists():
            files[uid], entries[uid], turns[uid] = (
                wav,
                channel_pipe(wav),
                read_textgrid(tg),
            )
    return files, entries, turns


def aishell4(split, root):
    turns = read_rttm(sorted((root / "TextGrid").glob("*.rttm")))
    files = {
        p.stem: p for p in sorted((root / "wav").glob("*.flac")) if p.stem in turns
    }
    return files, {u: channel_pipe(p) for u, p in files.items()}, turns


def voxconverse(split, audio, rttm_dir):
    turns = read_rttm(sorted(rttm_dir.glob("*.rttm")))
    files, entries = {}, {}
    if audio.suffix == ".zip":
        with zipfile.ZipFile(audio) as z:
            for name in z.namelist():
                stem = Path(name).stem
                if name.endswith(".wav") and "__MACOSX" not in name and stem in turns:
                    files[stem], entries[stem] = (
                        None,
                        f"unzip -p {shlex.quote(str(audio))} {shlex.quote(name)} |",
                    )
    else:
        for wav in sorted(audio.glob("*.wav")):
            if wav.stem in turns:
                files[wav.stem], entries[wav.stem] = wav, str(wav)
    return files, entries, turns


def earnings21(split, root):
    turns = read_rttm(sorted((root / "rttms").glob("*.rttm")))
    files = {
        p.stem: p for p in sorted((root / "media").glob("*.mp3")) if p.stem in turns
    }
    # some calls are stereo: score the first channel
    return files, {u: channel_pipe(p) for u, p in files.items()}, turns


def ava_speech(split, root):
    turns = {}
    with (root / "ava_speech_labels_v1.csv").open() as f:
        for vid, start, end, label in csv.reader(f):
            turns.setdefault(vid, [])
            if label != "NO_SPEECH":
                turns[vid].append((float(start) - 900, float(end) - 900))
    files = {
        p.stem: p for p in sorted((root / "audio").glob("*.flac")) if p.stem in turns
    }
    return files, {u: str(p) for u, p in files.items()}, turns


def ten_vad(split, root):
    files, turns = {}, {}
    for scv in sorted(root.glob("*.scv")):
        wav = scv.with_suffix(".wav")
        if not wav.exists():
            continue
        fields = scv.read_text().strip().split(",")[1:]  # <name>,start,end,label,...
        files[scv.stem] = wav
        turns[scv.stem] = [
            (float(fields[i]), float(fields[i + 1]))
            for i in range(0, len(fields) - 2, 3)
            if fields[i + 2].strip() == "1"
        ]
    return files, {u: str(p) for u, p in files.items()}, turns


CORPORA = {
    "ami_mh": (ami_mh, ("dev", "eval")),
    "alimeeting": (alimeeting, ("dev", "eval")),
    "aishell4": (aishell4, ("eval",)),
    "voxconverse": (voxconverse, ("dev", "eval")),
    "earnings21": (earnings21, ("eval",)),
    "ava_speech": (ava_speech, ("eval",)),
    "ten_vad": (ten_vad, ("eval",)),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("corpus", choices=sorted(CORPORA))
    parser.add_argument("split", choices=["dev", "eval"])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prep, splits = CORPORA[args.corpus]
    if args.split not in splits:
        raise SystemExit(f"{args.corpus} has no {args.split} split (only {splits})")
    files, entries, turns = prep(args.split, *[p.resolve() for p in args.paths])

    recordings, speech, skipped = {}, {}, []
    for uid, entry in entries.items():
        duration = None
        if files[uid] is not None:  # zip members have no path to probe
            info = readable_audio_info(files[uid], skipped)
            if info is None:
                continue
            duration = info[0]
        rec = f"{args.corpus}_{uid}"
        recordings[rec] = entry
        merged = merge_intervals(turns[uid])
        speech[rec] = clip_intervals(merged, duration) if duration else merged
    n = write_vad_dir(args.out_dir, recordings, speech)
    write_skipped(args.out_dir, skipped)
    logging.info("%s %s: %d recordings -> %s", args.corpus, args.split, n, args.out_dir)


if __name__ == "__main__":
    main()
