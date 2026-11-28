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
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_label_check import merge, textgrid_words  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad_data_utils import (  # noqa: E402
    format_tags,
    tag_only_categories,
    transcript_info,
)

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


def write_set(out: Path, recordings, reference: str, uem=None):
    """Write recordings [(rec_id, mono float32 at FS, segments[, categories])] as a set.

    segments: [(start, end)] s, written as one pseudo speaker "speech", or
    [(start, end, speaker)], written per speaker (for md-eval -1). uem: {rec_id:
    [(start, end)]}, written as the file uem (score_vad.py score --uem).
    categories: [(start, end, category, tags, layer)] s of a recording, written
    to the file categories (local/v2/category_ref.py), as vad_data_utils does.
    """
    from espnet2.fileio.omniio_audio import OmniIOAudioWriter

    out.mkdir(parents=True, exist_ok=True)
    nums, rttm, cats = {}, [], []
    with OmniIOAudioWriter(out / "audio", out / "wav.scp") as w:
        for rid, x, segs, *extra in recordings:
            for a, b, c, t, layer in sorted(extra[0] if extra else []):
                if b > a:
                    cats.append(f"{rid} {a:.3f} {b:.3f} {c} {format_tags(t)} {layer}\n")
            w[rid] = FS, x
            nums[rid] = len(x)
            dur = len(x) / FS
            by_spk = collections.defaultdict(list)
            for seg in segs:
                by_spk[seg[2] if len(seg) == 3 else "speech"].append(seg[:2])
            for spk, ss in sorted(by_spk.items()):
                for s, e in merge(ss):
                    s, e = max(0.0, s), min(dur, e)
                    if e > s:
                        rttm.append(
                            f"SPEAKER {rid} 1 {s:.3f} {e - s:.3f} "
                            f"<NA> <NA> {spk} <NA> <NA>\n"
                        )
    (out / "utt2num_samples").write_text(
        "".join(f"{r} {n}\n" for r, n in sorted(nums.items()))
    )
    (out / "rttm").write_text("".join(sorted(rttm)))
    if uem:
        (out / "uem").write_text(
            "".join(
                f"{r} 1 {s:.3f} {min(e, nums[r] / FS):.3f}\n"
                for r in sorted(uem)
                for s, e in uem[r]
            )
        )
    if cats:
        (out / "categories").write_text("".join(cats))
    (out / "feats_type").write_text("raw\n")
    (out / "audio_format").write_text("omniio\n")
    (out / "reference.txt").write_text(reference.strip() + "\n")
    speech = sum(e - s for r in nums for s, e in merge(rttm_spans(rttm, r)))
    total = sum(nums.values()) / FS
    print(
        f"{out}: {len(nums)} recordings, {total / 3600:.2f} h, speech "
        f"{speech / 3600:.2f} h ({100 * speech / max(total, 1e-9):.1f} %)",
        flush=True,
    )


def rttm_spans(lines, rid):
    out = []
    for x in lines:
        p = x.split()
        if p[1] == rid:
            out.append((float(p[3]), float(p[3]) + float(p[4])))
    return out


def hms(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def chime6(root: Path, refs: Path, split: str, kind: str):
    """CHiME-6 dev or eval: array U06, channel 1, one recording per session.

    kind "track2": the forced-alignment reference of the CHiME-6 Track 2 baseline
    (refs/<split>_rttm, per speaker) and its UEM (refs/uem_file, the U06 rows);
    kind "u06ann": the annotation (transcription JSON utterance times, the union
    of the speakers, as chime6_<split>_v1) over each whole recording.
    """
    fa = collections.defaultdict(list)
    for line in (refs / f"{split}_rttm").read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            s = float(p[3])
            fa[p[1].split("_")[0]].append((s, s + float(p[4]), p[7]))
    uem = {}
    for line in (refs / "uem_file").read_text().splitlines():
        p = line.split()
        if p and p[0].endswith("_U06"):
            uem[p[0].split("_")[0]] = (float(p[2]), float(p[3]))
    for session in sorted(fa):
        wav = sorted(root.glob(f"**/audio/{split}/{session}_U06.CH1.wav"))
        if not wav:
            raise FileNotFoundError(f"{session}_U06.CH1.wav under {root}")
        x, rate = soundfile.read(str(wav[0]), dtype="float32")
        x = to_mono(x, rate)
        rid = f"chime6_{split}_{session}_U06"
        if kind == "track2":
            yield rid, x, fa[session], [uem[session]]
        else:
            js = sorted(root.glob(f"**/transcriptions/{split}/{session}.json"))
            utts = json.loads(js[0].read_text(encoding="utf-8"))
            segs = [(hms(u["start_time"]), hms(u["end_time"])) for u in utts]
            yield rid, x, segs, None


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


def mfa_spn(archive: Path, subset: str):
    """{utterance id: [(start, end)]} of the "spn" (unknown word) intervals of MFA."""
    z = zipfile.ZipFile(archive)
    out = {}
    for n in z.namelist():
        if n.endswith(".TextGrid") and f"/{subset}/" in f"/{n}":
            uid = n.rsplit("/", 1)[-1][: -len(".TextGrid")]
            text = z.read(n).decode("utf-8", "replace")
            out[uid] = [
                (float(a), float(b))
                for a, b, w in INTERVAL_RE.findall(text)
                if w.strip().lower() == "spn"
            ]
    return out


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
    # unknown words (MFA "spn") are not in the reference, which leaves them
    # non-speech; they are speech/words details for the category references
    spn = mfa_spn(mfa, "test-clean") if mfa else {}
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
        segs, cats = [], []
        info = z.read(f"{s}/transcription/meeting_info.txt").decode()
        for line in info.splitlines()[1:]:
            f = line.split("\t")
            start, end, uid = float(f[0]), float(f[1]), f[3]
            if uid in words:
                segs += [(start + lag + a, start + lag + b) for a, b in words[uid]]
                cats += [
                    (start + lag + a, start + lag + b, "speech/words", ["spn"], 2)
                    for a, b in spn.get(uid, [])
                ]
                stats["utterances_mfa"] += 1
            else:
                segs.append((start + lag, end + lag))
                stats["utterances_no_mfa"] += 1
        rid = f"libricss_{cond}_session{m.group(1)}"
        print(f"{rid}: offset {lag * 1000:+.0f} ms, {len(x) / FS:.0f} s", flush=True)
        yield rid, x, merge(segs, bridge), cats
    print(f"LibriCSS: {dict(stats)}", flush=True)


def aishell5(root: Path):
    stats, markers = collections.Counter(), collections.Counter()
    for part in ("Eval1", "Eval2"):
        path = root / "aishell5/ARCHIVES" / f"{part}.tar.gz"
        speech = collections.defaultdict(list)  # session -> [(start, end)]
        # session -> marker-only intervals: speech/blank details (and a vocal
        # segment for a sound marker) for the category references
        cats = collections.defaultdict(list)
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
                            _, tags = transcript_info(w)
                            cats[name[-2]].append((s, e, "speech/blank", tags, 2))
                            cats[name[-2]] += tag_only_categories(s, e, tags)
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
                yield (
                    f"aishell5_{part}_{name[-2]}_{mm.group(1)}",
                    to_mono(x, rate),
                    speech[name[-2]],
                    cats[name[-2]],
                )
    print(
        f"AISHELL-5: {dict(stats)}; non-speech marker intervals (top 10): "
        f"{markers.most_common(10)}",
        flush=True,
    )


REFERENCES = {
    "chime6_dev_track2": """CHiME-6 dev (S02, S09), array U06 channel 1 (raw; the Track
2 baseline scores the WPE + BeamformIt output of U06). Reference: the forced-alignment
RTTM of the CHiME-6 Track 2 baseline (github.com/nateanl/chime6_rttm, MIT), per
speaker; scoring region: its UEM (kaldi egs/chime6/s5_track2/local/uem_file, U06
rows). CHiME-6 licence: evaluation only.""",
    "chime6_eval_track2": """CHiME-6 eval (S01, S21), array U06 channel 1 (raw; the
Track 2 baseline scores the WPE + BeamformIt output of U06). Reference: the
forced-alignment RTTM of the CHiME-6 Track 2 baseline (github.com/nateanl/chime6_rttm,
MIT), per
speaker; scoring region: its UEM (kaldi egs/chime6/s5_track2/local/uem_file, U06
rows). CHiME-6 licence: evaluation only.""",
    "chime6_dev_u06ann": """CHiME-6 dev (S02, S09), array U06 channel 1. Reference: the
transcription JSON utterance times, union of the speakers (as chime6_dev_v1), each
whole recording.""",
    "chime6_eval_u06ann": """CHiME-6 eval (S01, S21), array U06 channel 1. Reference:
the transcription JSON utterance times, union of the speakers (as chime6_eval_v1), each
whole recording.""",
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
    ap.add_argument("--chime6", type=Path, default=None, help="CHiME-6 root (db.sh)")
    ap.add_argument(
        "--chime6_refs",
        type=Path,
        default=None,
        help="dev_rttm, eval_rttm, uem_file (local/v2/downloads/chime6_track2.tsv)",
    )
    a = ap.parse_args()
    out = a.out / a.set
    if (out / ".done").exists():
        print(f"{out} is done; left as is")
        return
    if a.set.startswith("chime6_"):
        _, split, kind = a.set.split("_")
        items = list(chime6(a.chime6, a.chime6_refs, split, kind))
        uem = {r: u for r, _, _, u in items if u}
        recs = [(r, x, segs) for r, x, segs, _ in items]
        write_set(out, recs, REFERENCES[a.set], uem or None)
    else:
        recs = (
            libricss(a.root, a.mfa, a.bridge)
            if a.set == "libricss_eval"
            else aishell5(a.root)
        )
        write_set(out, recs, REFERENCES[a.set])
    (out / ".done").touch()


if __name__ == "__main__":
    main()
