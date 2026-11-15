#!/usr/bin/env python3
"""Standard VAD benchmarks as test sets of this recipe, read from their archives.

    python3 local/v2/bench_test_sets.py --set <name> --root ${VAD_CORPORA} \
        --out dump/raw \
        [--mfa ${VAD_CORPORA}/librispeech_mfa/ARCHIVES/librispeech_alignments.zip]

Writes dump/raw/<name> (marked done by .done; an unfinished one is written again):
the audio as an Omni-IO archive (audio/, wav.scp, 16 kHz mono), utt2num_samples,
rttm (one pseudo speaker "speech", seconds, md-eval format), feats_type,
audio_format, and reference.txt (where the audio and the reference come from, licence).
The archives are read in place; nothing is unpacked. Sets:

libricss_eval   LibriCSS (libricss/ARCHIVES/for_release.zip; recordings CC BY 4.0 from
                LibriSpeech, tools MIT): sessions 1-9 of every condition (session 0 is
                the dev set), channel 0 of record/raw_recording.wav, 54 recordings.
                Reference: the MFA word alignment of each played LibriSpeech
                utterance (--mfa, Lugosch's LibriSpeech alignments, CC BY 4.0) at its
                start time in transcription/meeting_info.txt, words with gaps under
                --bridge s joined; the utterance from start to end when MFA has no
                alignment of it. The offset of the recording against clean/mix.wav
                (cross-correlation of the first 120 s) is added to the times.
aishell5_eval   AISHELL-5 (aishell5/ARCHIVES/Eval1.tar.gz and Eval2.tar.gz, OpenSLR 159,
                CC BY-SA 4.0: evaluation only): the four far-field channels DX0kC01 of
                every session, each a recording. Reference: the manual transcription
                intervals of the session's near-field TextGrids (DA*.TextGrid, one
                tier per speaker), the union over the speakers; intervals without
                text, or with bracketed markers only, are not speech.
"""

import argparse
import collections
import io
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_label_check import merge, textgrid_words  # noqa: E402

FS = 16000
INTERVAL_RE = re.compile(r'xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = "([^"]*)"')
MARKER_RE = re.compile(r"^(\s*([<\[][^>\]]*[>\]])\s*)*$")


def textgrid_intervals(text: str):
    """[(start, end, text)] of every interval of every tier of a TextGrid."""
    out = []
    for tier in re.split(r"\n\s*item \[\d+\]:", text)[1:]:
        out += [(float(a), float(b), t) for a, b, t in INTERVAL_RE.findall(tier)]
    return out


def is_speech(text: str) -> bool:
    """Speech unless the interval text is empty or bracketed markers only."""
    return bool(text.strip()) and not MARKER_RE.match(text)


def write_set(out: Path, recordings, reference: str):
    """Write recordings [(rec_id, mono float32 at FS, [(start, end)] s)] as a set."""
    from espnet2.fileio.omniio_audio import OmniIOAudioWriter

    out.mkdir(parents=True, exist_ok=True)
    nums, rttm = {}, []
    with OmniIOAudioWriter(out / "audio", out / "wav.scp") as w:
        for rid, x, segs in recordings:
            w[rid] = FS, x
            nums[rid] = len(x)
            dur = len(x) / FS
            for s, e in merge(segs):
                s, e = max(0.0, s), min(dur, e)
                if e > s:
                    rttm.append(
                        f"SPEAKER {rid} 1 {s:.3f} {e - s:.3f} "
                        "<NA> <NA> speech <NA> <NA>\n"
                    )
    (out / "utt2num_samples").write_text(
        "".join(f"{r} {n}\n" for r, n in sorted(nums.items()))
    )
    (out / "rttm").write_text("".join(sorted(rttm)))
    (out / "feats_type").write_text("raw\n")
    (out / "audio_format").write_text("omniio\n")
    (out / "reference.txt").write_text(reference.strip() + "\n")
    speech = sum(float(x.split()[4]) for x in rttm)
    total = sum(nums.values()) / FS
    print(
        f"{out}: {len(nums)} recordings, {total / 3600:.2f} h, speech "
        f"{speech / 3600:.2f} h ({100 * speech / max(total, 1e-9):.1f} %)",
        flush=True,
    )


def to_mono(x, rate, channel=0):
    if x.ndim == 2:
        x = x[:, channel]
    if rate != FS:
        raise ValueError(f"expected {FS} Hz, got {rate}")
    return np.asarray(x, dtype=np.float32)


def lag_seconds(ref, rec, seconds=120.0):
    """Delay (s) of `rec` against `ref` that maximizes their cross-correlation."""
    from scipy.signal import correlate

    n = int(seconds * FS)
    a, b = ref[:n] - ref[:n].mean(), rec[:n] - rec[:n].mean()
    c = correlate(b, a, mode="full", method="fft")
    return (int(np.argmax(np.abs(c))) - (len(a) - 1)) / FS


def mfa_words(archive: Path, subset: str):
    """{utterance id: [(start, end)]} of the MFA words of a LibriSpeech subset."""
    z = zipfile.ZipFile(archive)
    out = {}
    for n in z.namelist():
        if n.endswith(".TextGrid") and f"/{subset}/" in f"/{n}":
            uid = n.rsplit("/", 1)[-1][: -len(".TextGrid")]
            out[uid] = textgrid_words(z.read(n).decode("utf-8", "replace"))
    return out


def libricss(root: Path, mfa: Path, bridge: float):
    words = mfa_words(mfa, "test-clean") if mfa else {}
    print(f"MFA: {len(words)} test-clean utterances", flush=True)
    z = zipfile.ZipFile(root / "libricss/ARCHIVES/for_release.zip")
    sessions = sorted(
        {
            n.split("/transcription/")[0]
            for n in z.namelist()
            if n.endswith("/transcription/meeting_info.txt")
        }
    )
    stats = collections.Counter()
    for s in sessions:
        cond, name = s.split("/")[-2], s.split("/")[-1]
        m = re.search(r"_session(\d+)_", name)
        if not m or m.group(1) == "0":
            continue  # session 0 of every condition is the dev set
        rec, rate = soundfile.read(io.BytesIO(z.read(f"{s}/record/raw_recording.wav")))
        x = to_mono(rec, rate, 0)
        mix, rate = soundfile.read(io.BytesIO(z.read(f"{s}/clean/mix.wav")))
        lag = lag_seconds(to_mono(mix, rate), x)
        segs = []
        info = z.read(f"{s}/transcription/meeting_info.txt").decode()
        for line in info.splitlines()[1:]:
            f = line.split("\t")
            start, end, uid = float(f[0]), float(f[1]), f[3]
            if uid in words:
                segs += [(start + lag + a, start + lag + b) for a, b in words[uid]]
                stats["utterances_mfa"] += 1
            else:
                segs.append((start + lag, end + lag))
                stats["utterances_no_mfa"] += 1
        rid = f"libricss_{cond}_session{m.group(1)}"
        print(f"{rid}: offset {lag * 1000:+.0f} ms, {len(x) / FS:.0f} s", flush=True)
        yield rid, x, merge(segs, bridge)
    print(f"LibriCSS: {dict(stats)}", flush=True)


def aishell5(root: Path):
    stats, markers = collections.Counter(), collections.Counter()
    for part in ("Eval1", "Eval2"):
        path = root / "aishell5/ARCHIVES" / f"{part}.tar.gz"
        speech = collections.defaultdict(list)  # session -> [(start, end)]
        with tarfile.open(path, "r|gz") as t:  # pass 1: the near-field TextGrids
            for m in t:
                name = m.name.split("/")
                if m.isfile() and re.match(r"DA\d+\.TextGrid$", name[-1]):
                    text = t.extractfile(m).read().decode("utf-8", "replace")
                    for s, e, w in textgrid_intervals(text):
                        if is_speech(w):
                            speech[name[-2]].append((s, e))
                        elif w.strip():
                            markers[w.strip()] += 1
        with tarfile.open(path, "r|gz") as t:  # pass 2: the far-field channels
            for m in t:
                name = m.name.split("/")
                mm = re.match(r"(DX0[1-4])C01\.wav$", name[-1])
                if not (m.isfile() and mm):
                    continue
                if name[-2] not in speech:
                    stats["far_field_without_reference"] += 1
                    continue
                x, rate = soundfile.read(io.BytesIO(t.extractfile(m).read()))
                stats["recordings"] += 1
                yield f"aishell5_{part}_{name[-2]}_{mm.group(1)}", to_mono(
                    x, rate
                ), speech[name[-2]]
    print(
        f"AISHELL-5: {dict(stats)}; non-speech marker intervals (top 10): "
        f"{markers.most_common(10)}",
        flush=True,
    )


REFERENCES = {
    "libricss_eval": """LibriCSS eval (sessions 1-9 of 0L, 0S, OV10-OV40), channel 0 of
the recordings in libricss/ARCHIVES/for_release.zip. Licence: CC BY 4.0 (LibriSpeech);
LibriCSS tools MIT. Reference: MFA word alignments (Lugosch, CC BY 4.0) of the played
LibriSpeech utterances at their meeting_info.txt start times plus the recording's
offset against clean/mix.wav; gaps under 0.2 s joined.""",
    "aishell5_eval": """AISHELL-5 Eval1 + Eval2 (OpenSLR 159), the four far-field
channels of every session. Licence: CC BY-SA 4.0, evaluation only (never training).
Reference: manual transcription intervals of the near-field TextGrids (one tier per
speaker), union over the speakers; empty or bracketed-marker-only intervals are not
speech.""",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", required=True, choices=sorted(REFERENCES))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("dump/raw"), type=Path)
    ap.add_argument("--mfa", type=Path, default=None)
    ap.add_argument("--bridge", type=float, default=0.2)
    a = ap.parse_args()
    out = a.out / a.set
    if (out / ".done").exists():
        print(f"{out} is done; left as is")
        return
    recs = (
        libricss(a.root, a.mfa, a.bridge)
        if a.set == "libricss_eval"
        else aishell5(a.root)
    )
    write_set(out, recs, REFERENCES[a.set])
    (out / ".done").touch()


if __name__ == "__main__":
    main()
