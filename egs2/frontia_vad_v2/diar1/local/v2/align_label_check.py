#!/usr/bin/env python3
"""Compare CTC alignment labels (as stage 4 makes them) with human labels.

    python3 local/v2/align_label_check.py align --set ami --channel Mix-Headset \
        --dump <dump of ami_dev> --annotations <ami_public_manual zip> --out <dir>
    python3 local/v2/align_label_check.py score --out <dir> --ref <rttm> \
        [--label_set forced_align]

align: every speaker's words of the transcript (NXT words XML of AMI or ICSI)
are grouped into utterances (a new one after a gap of more than --utt_gap s
between timed words); each utterance is cut from the recording with --window s
on each side and aligned to its words with OWSM-CTC through ctc_batch.py,
with the settings of stage 4 (local/data.sh). Writes words.tsv (per word: the
CTC segmentation segment and score, the token span, and the torchaudio forced
alignment span and score, with whether --min_word_score would drop the
utterance), xml_words.tsv (the transcript's own word times), dropped.tsv,
fa_dropped.tsv and durations.tsv; --bench_longest also times forced
alignment of the longest recording as a whole.

score: for each label set (segmentation, tokens, forced_align, xml_words),
every aligned word without a score filter, the start pad x end pad x
min_pause grid of frame-level (10 ms) DetER against the reference RTTM over
each whole recording; the best setting with its miss and false alarm, onset
and offset deltas and pause distribution. A best setting on the largest
value of a grid dimension widens that dimension once (reported). Also the
words that --min_word_score thresholds would drop from forced_align.
"""

import argparse
import collections
import html
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

W_RE = re.compile(r"<w\b([^>]*)>(.*?)</w>", re.S)
ATTR_RE = re.compile(r'([\w:]+)="([^"]*)"')


def merge(spans, min_pause=0.0):
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] < min_pause or out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def read_words(annotations: Path, meeting: str, kind: str):
    """{speaker: [(start or None, end or None, token)]} of one meeting."""
    files = {}
    if annotations.suffix == ".zip":
        z = zipfile.ZipFile(annotations)
        for n in z.namelist():
            m = re.match(rf".*words/{meeting}\.(\w+)\.words\.xml$", n)
            if m:
                files[m[1]] = z.read(n).decode("iso-8859-1")
    else:
        for p in annotations.glob(f"{meeting}.*.words.xml"):
            files[p.name.split(".")[1]] = p.read_text(encoding="iso-8859-1")
    out = {}
    for spk, text in files.items():
        words = []
        for attrs, tok in W_RE.findall(text):
            a = dict(ATTR_RE.findall(attrs))
            if a.get("punc") == "true":
                continue
            if kind == "icsi" and a.get("c", "W") != "W":
                continue
            tok = html.unescape(tok).strip()
            if not tok:
                continue
            s = float(a["starttime"]) if a.get("starttime") else None
            e = float(a["endtime"]) if a.get("endtime") else None
            if tok.startswith("'") and words:  # ICSI: "We" "'re" -> "We're"
                ps, pe, pt = words[-1]
                words[-1] = (ps, e if e is not None else pe, pt + tok)
                continue
            words.append((s, e, tok))
        out[spk] = words
    return out


def utterances(words, gap):
    """[(start, end, [tokens], [(s, e) of timed words])] of one speaker."""
    utts, cur = [], None
    last_end = None
    for s, e, tok in words:
        t0 = s if s is not None else e
        if (
            cur is not None
            and t0 is not None
            and last_end is not None
            and t0 - last_end > gap
        ):
            utts.append(cur)
            cur = None
        if cur is None:
            cur = [None, None, [], []]
        cur[2].append(tok)
        if s is not None and e is not None and e >= s:
            cur[3].append((s, e))
        for t in (s, e):
            if t is not None:
                cur[0] = t if cur[0] is None else min(cur[0], t)
                cur[1] = t if cur[1] is None else max(cur[1], t)
        if e is not None:
            last_end = e
        elif s is not None:
            last_end = s
    if cur is not None:
        utts.append(cur)
    return [u for u in utts if u[0] is not None and u[1] > u[0]]


def bench_longest(a, aligner, audio, scp, model_words, align_batched):
    """Time and GPU memory of forced alignment of the longest recording whole.

    All its transcript words in time order as one sequence (the alignment
    itself is not scored: overlapping speakers make it meaningless).
    """
    import torch

    lengths = {rid: len(audio[rid][1]) for rid in scp}
    rid = max(lengths, key=lengths.get)
    meeting = rid.split("_")[1]
    words = []
    for spk, ws in read_words(Path(a.annotations), meeting, a.set).items():
        words += [(s if s is not None else e, tok) for s, e, tok in ws if (s or e)]
    toks = [t for _, t in sorted(words)]
    rate, x = audio[rid]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    res = align_batched(
        aligner,
        iter([(rid, model_words(toks), "<eng>", lambda: (rate, x))]),
        a.batch_seconds,
        a.workers,
        64,
        forced_align=True,
    )
    wall = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    seg, spans, err = res[rid]
    fa = None if err else spans["forced_align"]
    if err:
        status = "error " + str(err)
    else:
        status = "too few frames" if isinstance(fa, str) else "ok"
    print(
        f"BENCH longest {rid}: {len(x) / rate / 60:.1f} min, {len(toks)} words; "
        f"whole pipeline {wall:.0f} s, peak GPU memory above the model "
        f"{peak:.1f} GiB; forced_align "
        f"{status}",
        flush=True,
    )


MFA_SILENCE = {"", "sil", "sp", "spn", "<eps>"}


def textgrid_words(text: str):
    """[(start, end)] of the non-silent intervals of a TextGrid's "words" tier.

    Praat TextGrid long text format, as the Montreal Forced Aligner writes it.
    """
    tiers = re.split(r"\n\s*item \[\d+\]:", text)
    for tier in tiers[1:]:
        name = re.search(r'name = "([^"]*)"', tier)
        if not name or name[1] != "words":
            continue
        out = []
        for xmin, xmax, word in re.findall(
            r'xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = "([^"]*)"', tier
        ):
            if word.strip().lower() not in MFA_SILENCE:
                out.append((float(xmin), float(xmax)))
        return out
    raise ValueError("no words tier")


def read_mfa_words(archive: Path):
    """{utterance id: [(start, end)]} of MFA word alignments in a zip.

    The zip holds TextGrids <...>/<utterance id>.TextGrid.
    """
    z = zipfile.ZipFile(archive)
    out = {}
    for n in z.namelist():
        if n.endswith(".TextGrid") and ("dev-clean" in n or "dev-other" in n):
            uid = n.rsplit("/", 1)[-1][: -len(".TextGrid")]
            out[uid] = textgrid_words(z.read(n).decode("utf-8", "replace"))
    if not out:
        raise RuntimeError(f"no dev TextGrid in {archive}")
    return out


def librispeech_utterances(root: Path, subsets):
    """(subset, utterance id, flac path, words) of LibriSpeech subsets."""
    for sub in subsets:
        for trans in sorted((root / sub).glob("*/*/*.trans.txt")):
            for line in trans.read_text().splitlines():
                uid, text = line.split(" ", 1)
                yield sub, uid, trans.parent / f"{uid}.flac", text.split()


def mfa_rttm(words_by_utt, durs, bridge: float):
    """RTTM lines of the MFA words.

    Each word from its start to its end, gaps shorter than `bridge` s joined
    (the convention of the test RTTMs).
    """
    lines = []
    for uid, words in sorted(words_by_utt.items()):
        for s, e in merge(words, bridge):
            lines.append(f"SPEAKER {uid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {uid} <NA>\n")
    return lines


def mfa_pause_report(words_by_utt, bridge: float):
    """Pauses inside the utterances of the MFA reference (after bridging)."""
    pauses, n_utt = [], 0
    has = collections.Counter()
    for words in words_by_utt.values():
        segs = merge(words, bridge)
        g = gaps(segs)
        pauses += g
        n_utt += 1
        for t in (0.2, 0.5, 1.0):
            has[t] += any(x >= t for x in g)
    g = np.array(pauses)
    q = (
        " / ".join(f"{np.percentile(g, p):.2f}" for p in (10, 50, 90))
        if len(g)
        else "-"
    )
    return (
        f"MFA reference: {n_utt} utterances, {len(g)} pauses inside them "
        f"(p10/50/90 {q} s, {g.sum() / 3600:.2f} h); utterances with a pause "
        + ", ".join(
            f">={t:g} s {100 * has[t] / max(1, n_utt):.1f} %" for t in (0.2, 0.5, 1.0)
        )
    )


def librispeech_items(a, model_words, out: Path):
    """Alignment items and MFA reference words of LibriSpeech utterances.

    One clip per recording, no packing; writes mfa.rttm.
    """
    import soundfile

    mfa = read_mfa_words(Path(a.annotations))
    items, meta, xml_rows, durs = [], {}, [], []
    words_by_utt, missing = {}, 0
    for sub, uid, path, words in librispeech_utterances(Path(a.dump), a.subsets):
        if uid not in mfa:
            missing += 1
            continue
        durs.append(f"{uid}\t{soundfile.info(str(path)).duration:.3f}\n")
        meta[uid] = (uid, 0.0, words)
        words_by_utt[uid] = mfa[uid]
        xml_rows += [f"{uid}\t{uid}\t{s:.3f}\t{e:.3f}\n" for s, e in mfa[uid]]

        def read(p=path):
            x, rate = soundfile.read(str(p), dtype="float32")
            return rate, x

        items.append((uid, model_words(words), "<eng>", read))
    (out / "mfa.rttm").write_text("".join(mfa_rttm(words_by_utt, None, a.bridge)))
    print(f"utterances without MFA words (left out): {missing}")
    print(mfa_pause_report(words_by_utt, a.bridge), flush=True)
    return items, meta, xml_rows, durs


def cmd_align(a):
    sys.path.insert(0, str(Path(a.v2).resolve()))
    import torch
    from align import absolute_config, model_files, model_words
    from ctc_batch import align_batched

    from espnet2.bin.s2t_align import CTCSegmentation
    from espnet2.fileio.omniio_audio import OmniIOAudioScpReader

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scp, audio = {}, None
    if a.set != "librispeech":
        for line in open(Path(a.dump) / "wav.scp"):
            rid = line.split(maxsplit=1)[0]
            if rid.endswith("_" + a.channel):
                scp[rid] = line
        (out / "wav.scp").write_text("".join(scp[r] for r in sorted(scp)))
        audio = OmniIOAudioScpReader(out / "wav.scp")
    root, config, model = model_files(a.model_tag, a.revision)
    torch.manual_seed(0)
    config = absolute_config(root, config, out / "owsm_config.yaml", a.use_sdpa)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    aligner = CTCSegmentation(
        s2t_train_config=config,
        s2t_model_file=model,
        device="cuda",
        dtype=a.dtype,
        kaldi_style_text=False,
        time_stamps="auto",
    )
    items, meta, xml_rows, durs = [], {}, [], []
    cache = {}

    def reader(rid, s, e):
        def read():
            if rid not in cache:
                cache.clear()
                rate, x = audio[rid]
                cache[rid] = (rate, np.asarray(x, dtype=np.float32))
            rate, x = cache[rid]
            return rate, x[int(s * rate) : int(e * rate)]

        return read

    if a.set == "librispeech":
        items, meta, xml_rows, durs = librispeech_items(a, model_words, out)
    for rid in sorted(scp):
        meeting = rid.split("_")[1]
        rate, x = audio[rid]
        dur = len(x) / rate
        durs.append(f"{rid}\t{dur:.3f}\n")
        for spk, words in read_words(Path(a.annotations), meeting, a.set).items():
            for k, (us, ue, toks, timed) in enumerate(utterances(words, a.utt_gap)):
                uid = f"{rid}__{spk}{k:05d}"
                ws, we = max(0.0, us - a.window), min(dur, ue + a.window)
                meta[uid] = (rid, ws, toks)
                for s, e in timed:
                    xml_rows.append(f"{rid}\t{uid}\t{s:.3f}\t{e:.3f}\n")
                items.append((uid, model_words(toks), "<eng>", reader(rid, ws, we)))
    print(f"{len(scp) or len(durs)} recordings, {len(items)} utterances", flush=True)
    results = align_batched(
        aligner, iter(items), a.batch_seconds, a.workers, 64, forced_align=True
    )
    rows, dropped, fa_dropped, fa_too_few = [], [], [], 0
    for uid, (rid, ws, toks) in meta.items():
        segments, spans, error = results.get(uid, (None, None, "missing"))
        if error is not None:
            dropped.append(f"{uid}\t{len(toks)}\talignment failed: {error}\n")
            continue
        worst = min(c for _, _, c in segments)
        reason = f"word score {worst:.2f}" if worst < a.min_word_score else ""
        if reason:
            dropped.append(f"{uid}\t{len(toks)}\t{reason}\n")
        fa = spans["forced_align"]
        if isinstance(fa, str):
            fa_too_few += 1
            fa_dropped.append(f"{uid}\t{len(toks)}\t{fa}\n")
            fa = [(float("nan"), float("nan"), float("-inf"))] * len(toks)
            fa_reason = "x"
        else:
            fa_worst = min(c for _, _, c in fa)
            fa_reason = (
                f"word score {fa_worst:.2f}" if fa_worst < a.min_word_score else ""
            )
            if fa_reason:
                fa_dropped.append(f"{uid}\t{len(toks)}\t{fa_reason}\n")
        for tok, (s, e, c), (ts, te), (fs, fe, fc) in zip(
            toks, segments, spans["tokens"], fa
        ):
            rows.append(
                f"{rid}\t{uid}\t{ws + s:.3f}\t{ws + e:.3f}\t{c:.3f}"
                f"\t{int(bool(reason))}\t{ws + ts:.3f}\t{ws + te:.3f}"
                f"\t{ws + fs:.3f}\t{ws + fe:.3f}\t{fc:.3f}"
                f"\t{int(bool(fa_reason))}\n"
            )
    (out / "words.tsv").write_text(
        "rid\tuid\tstart\tend\tscore\tdropped\ttok_start\ttok_end"
        "\tfa_start\tfa_end\tfa_score\tfa_dropped\n" + "".join(rows)
    )
    (out / "fa_dropped.tsv").write_text("".join(fa_dropped))
    print(
        f"forced_align: too few frames in {fa_too_few} utterances, "
        f"{len(fa_dropped) - fa_too_few} below min_word_score",
        flush=True,
    )
    if a.bench_longest:
        bench_longest(a, aligner, audio, scp, model_words, align_batched)
    (out / "xml_words.tsv").write_text("rid\tuid\tstart\tend\n" + "".join(xml_rows))
    (out / "dropped.tsv").write_text("".join(dropped))
    (out / "durations.tsv").write_text("".join(durs))
    print(f"aligned {len(meta) - len(dropped)}, dropped {len(dropped)}", flush=True)


STEP = 0.01  # seconds per scoring frame


def frames(spans, n):
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[max(0, int(round(s / STEP))) : max(0, int(round(e / STEP)))] = True
    return m


def gaps(segs):
    return [b[0] - a[1] for a, b in zip(segs, segs[1:])]


def labels(word_rows, min_pause, pad_start, pad_end):
    """{rid: segments} of word rows, padded and merged.

    Per utterance the union of its word spans with pauses shorter than
    min_pause bridged (as align.py), each segment widened by pad_start before
    and pad_end after, then the union per recording.
    """
    by_utt = collections.defaultdict(list)
    for rid, uid, s, e in word_rows:
        by_utt[(rid, uid)].append((s, e))
    per_rec = collections.defaultdict(list)
    for (rid, _), spans in by_utt.items():
        for s, e in merge(spans, min_pause):
            per_rec[rid].append((max(0.0, s - pad_start), e + pad_end))
    return {r: merge(v) for r, v in per_rec.items()}


class Scorer:
    """Frame-level miss and false alarm against a reference, whole recordings."""

    def __init__(self, ref, durs):
        self.durs = durs
        self.n = {r: int(round(d / STEP)) for r, d in durs.items()}
        self.ref = {r: frames(ref.get(r, []), self.n[r]) for r in durs}
        self.refs = sum(int(m.sum()) for m in self.ref.values())

    def __call__(self, hyp):
        miss = fa = hyps = 0
        for r, ref in self.ref.items():
            h = frames(hyp.get(r, []), self.n[r])
            miss += int((ref & ~h).sum())
            fa += int((~ref & h).sum())
            hyps += int(h.sum())
        return dict(
            miss=100 * miss / self.refs,
            fa=100 * fa / self.refs,
            deter=100 * (miss + fa) / self.refs,
            ref_h=self.refs * STEP / 3600,
            hyp_h=hyps * STEP / 3600,
        )


def edges(hyp, ref):
    """Onset/offset deltas (label - reference) and uncovered reference speech.

    Deltas of the reference segments that labels overlap, and the share of
    reference speech no label overlaps.
    """
    on, off, uncovered, total = [], [], 0.0, 0.0
    for rid, rsegs in ref.items():
        h = hyp.get(rid, [])
        for rs, re_ in rsegs:
            total += re_ - rs
            ov = [(s, e) for s, e in h if e > rs and s < re_]
            if not ov:
                uncovered += re_ - rs
                continue
            on.append(ov[0][0] - rs)
            off.append(ov[-1][1] - re_)

    def pct(v):
        return " / ".join(f"{np.percentile(v, q) * 1000:+.0f}" for q in (10, 50, 90))

    return pct(on), pct(off), 100 * uncovered / total


def pause_stats(segs_by_rec):
    g = np.array([x for segs in segs_by_rec.values() for x in gaps(segs)])
    if not len(g):
        return "no pauses"
    return (
        f"{len(g)} pauses, median {np.median(g):.2f} s, "
        f"<0.2 s {100 * (g < 0.2).mean():.0f} %, "
        f"<0.5 s {100 * (g < 0.5).mean():.0f} %, "
        f"<1 s {100 * (g < 1.0).mean():.0f} %, <2 s {100 * (g < 2.0).mean():.0f} %"
    )


def label_sets(out: Path):
    """{name: word rows (rid, uid, start, end)} of an align output directory.

    Every aligned word, without any score filter.
    """
    table = [x.rstrip("\n").split("\t") for x in list(open(out / "words.tsv"))[1:]]
    sets = {"segmentation": [(f[0], f[1], float(f[2]), float(f[3])) for f in table]}
    if table and len(table[0]) > 7:
        sets["tokens"] = [(f[0], f[1], float(f[6]), float(f[7])) for f in table]
    if table and len(table[0]) > 11:
        sets["forced_align"] = [
            (f[0], f[1], float(f[8]), float(f[9]))
            for f in table
            if f[8] != "nan"  # too few frames: no forced alignment
        ]
    sets["xml_words"] = [
        (f[0], f[1], float(f[2]), float(f[3]))
        for f in (
            x.rstrip("\n").split("\t") for x in list(open(out / "xml_words.tsv"))[1:]
        )
    ]
    return sets, table


def score_filter(table, thresholds):
    """Words that min_word_score would drop from forced_align, per threshold.

    The words of utterances whose lowest forced-alignment word score is below
    the threshold.
    """
    lowest, words = {}, collections.Counter()
    for f in table:
        if len(f) > 11 and f[8] != "nan":
            lowest[f[1]] = min(lowest.get(f[1], 0.0), float(f[10]))
            words[f[1]] += 1
    total = sum(words.values())
    return [
        (
            t,
            sum(1 for u in lowest if lowest[u] < t),
            sum(words[u] for u in lowest if lowest[u] < t),
            total,
        )
        for t in thresholds
    ]


def grid_search(rows, scorer, starts, ends, pauses):
    """[(deter, miss, fa, pad_start, pad_end, min_pause)], best first."""
    out = []
    for mp in pauses:
        for ps in starts:
            for pe in ends:
                ev = scorer(labels(rows, mp, ps, pe))
                out.append((ev["deter"], ev["miss"], ev["fa"], ps, pe, mp))
    return sorted(out)


def widen(values, best, step):
    """values extended by two steps when best is the largest of them."""
    if best == max(values):
        return sorted(
            set(values)
            | {round(max(values) + step, 3), round(max(values) + 2 * step, 3)}
        )
    return values


def labelled_segments(word_rows, min_pause, pad_start, pad_end):
    """{rid: [(start, end, {utterance ids})]}: labels() with their utterances.

    Each merged segment keeps the transcript utterances it comes from.
    """
    by_utt = collections.defaultdict(list)
    for rid, uid, s, e in word_rows:
        by_utt[(rid, uid)].append((s, e))
    per_rec = collections.defaultdict(list)
    for (rid, uid), spans in by_utt.items():
        for s, e in merge(spans, min_pause):
            per_rec[rid].append((max(0.0, s - pad_start), e + pad_end, uid))
    out = {}
    for rid, segs in per_rec.items():
        merged = []
        for s, e, uid in sorted(segs):
            if merged and s <= merged[-1][1]:
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], e),
                    merged[-1][2] | {uid},
                )
            else:
                merged.append((s, e, {uid}))
        out[rid] = merged
    return out


def split_pauses(segs):
    """(pauses inside one utterance, pauses between utterances) of segments.

    A pause is inside when both neighbouring segments share an utterance.
    """
    inside, between = [], []
    for segs_ in segs.values():
        for (s1, e1, u1), (s2, e2, u2) in zip(segs_, segs_[1:]):
            (inside if u1 & u2 else between).append(s2 - e1)
    return inside, between


def utterance_spans(xml_rows):
    """{rid: [(start, end)]} of the transcript utterances (first to last word)."""
    by = collections.defaultdict(list)
    for rid, uid, s, e in xml_rows:
        by[(rid, uid)].append((s, e))
    out = collections.defaultdict(list)
    for (rid, _), w in by.items():
        out[rid].append((min(s for s, _ in w), max(e for _, e in w)))
    return out


def split_reference_pauses(refr, utts):
    """(pauses inside a transcript utterance, other pauses) of the reference.

    A pause is inside when some utterance of the recording spans it whole.
    """
    inside, between = [], []
    for rid, segs in refr.items():
        spans = utts.get(rid, [])
        for (s1, e1), (s2, e2) in zip(segs, segs[1:]):
            hit = any(us <= e1 and s2 <= ue for us, ue in spans)
            (inside if hit else between).append(s2 - e1)
    return inside, between


def describe(g):
    g = np.array(g)
    if not len(g):
        return "0"
    q = " / ".join(f"{np.percentile(g, p):.2f}" for p in (10, 50, 90))
    return f"{len(g)} (p10/50/90 {q} s, total {g.sum() / 3600:.2f} h)"


def cmd_pauses(a):
    """Where the pauses of a label set come from, and min_pause = infinity."""
    ref = collections.defaultdict(list)
    for line in open(a.ref):
        p = line.split()
        if p and p[0] == "SPEAKER":
            ref[p[1]].append((float(p[3]), float(p[3]) + float(p[4])))
    out = Path(a.out)
    durs = {
        x.split("\t")[0]: float(x.split("\t")[1]) for x in open(out / "durations.tsv")
    }
    refr = {r: merge(ref[r]) for r in durs}
    scorer = Scorer(refr, durs)
    sets, _ = label_sets(out)
    rows = sets[a.label_set]
    segs = labelled_segments(rows, a.min_pause, a.pad_start, a.pad_end)
    inside, between = split_pauses(segs)
    print(
        f"== {out.name} {a.label_set} at start pad {a.pad_start * 1000:.0f} ms, "
        "end pad "
        f"{a.pad_end * 1000:.0f} ms, min_pause {a.min_pause:g} s: pauses inside one "
        f"utterance {describe(inside)}; between utterances {describe(between)}"
    )
    rin, rbe = split_reference_pauses(refr, utterance_spans(sets["xml_words"]))
    print(
        f"   reference: pauses inside a transcript utterance {describe(rin)}; "
        f"outside {describe(rbe)}"
    )
    ev = scorer(labels(rows, a.min_pause, a.pad_start, a.pad_end))
    print(
        f"   DetER at that setting {ev['deter']:.2f} (miss {ev['miss']:.2f}, "
        f"FA {ev['fa']:.2f})"
    )
    for name in [n for n in sets if n in a.inf_sets]:
        res = grid_search(sets[name], scorer, a.starts, a.ends, [float("inf")])
        d, miss, fa, ps, pe, _ = res[0]
        print(
            f"   {name} min_pause infinity (each utterance first to last word): best "
            f"start pad {ps * 1000:.0f} ms, end pad {pe * 1000:.0f} ms -> "
            f"DetER {d:.2f} "
            f"(miss {miss:.2f}, FA {fa:.2f})",
            flush=True,
        )


def cmd_score(a):
    ref = collections.defaultdict(list)
    for line in open(a.ref):
        p = line.split()
        if p and p[0] == "SPEAKER":
            ref[p[1]].append((float(p[3]), float(p[3]) + float(p[4])))
    out = Path(a.out)
    durs = {
        x.split("\t")[0]: float(x.split("\t")[1]) for x in open(out / "durations.tsv")
    }
    refr = {r: merge(ref[r]) for r in durs}
    scorer = Scorer(refr, durs)
    sets, table = label_sets(out)
    print(
        f"== {out.name}: {len(durs)} recordings, "
        f"reference {scorer.refs * STEP / 3600:.2f} h; "
        f"reference {pause_stats(refr)}",
        flush=True,
    )
    if "forced_align" in sets and a.label_set in (None, "forced_align"):
        for t, nu, nw, total in score_filter(table, a.thresholds):
            print(
                f"  forced_align min_word_score {t:g}: would drop {nu} utterances, "
                f"{nw} of {total} words ({100 * nw / max(1, total):.1f} %)"
            )
        print(
            f"  forced_align: {sum(len(f) > 11 and f[8] == 'nan' for f in table)} "
            "word rows "
            "of utterances with too few frames (not scored)"
        )
    names = [a.label_set] if a.label_set else list(sets)
    for name in names:
        if name not in sets:
            print(f"  {name}: not in {out}")
            continue
        rows = sets[name]
        starts, ends, pauses = list(a.starts), list(a.ends), list(a.min_pauses)
        res = grid_search(rows, scorer, starts, ends, pauses)
        d, miss, fa, ps, pe, mp = res[0]
        widened = []
        for label, vals, best, step in (
            ("start pad", starts, ps, 0.05),
            ("end pad", ends, pe, 0.1),
            ("min_pause", pauses, mp, 1.0),
        ):
            if best == max(vals):
                widened.append(label)
        if a.print_grid:
            for r in sorted(res, key=lambda r: (r[5], r[3], r[4])):
                print(
                    f"  {name}: GRID start {r[3] * 1000:.0f} ms, "
                    f"end {r[4] * 1000:.0f} ms, "
                    f"min_pause {r[5]:g} s: DetER {r[0]:.2f} (miss {r[1]:.2f}, "
                    f"FA {r[2]:.2f})"
                )
        if a.no_widen:
            widened = []
        if widened:
            starts = widen(starts, ps, 0.05)
            ends = widen(ends, pe, 0.1)
            pauses = widen(pauses, mp, 1.0)
            res = grid_search(rows, scorer, starts, ends, pauses)
            d, miss, fa, ps, pe, mp = res[0]
        on_edge = [
            label
            for label, vals, best in (
                ("start pad", starts, ps),
                ("end pad", ends, pe),
                ("min_pause", pauses, mp),
            )
            if best == max(vals)
        ]
        hyp = labels(rows, mp, ps, pe)
        ev = scorer(hyp)
        on, off, unc = edges(hyp, refr)
        raw = scorer(labels(rows, 0.2, 0.0, 0.0))
        print(
            f"  {name}: no pad, min_pause 0.2: DetER {raw['deter']:.2f} "
            f"(miss {raw['miss']:.2f}, FA {raw['fa']:.2f}), labels {raw['hyp_h']:.2f} h"
        )
        print(
            f"  {name}: BEST start pad {ps * 1000:.0f} ms, end pad {pe * 1000:.0f} ms, "
            f"min_pause {mp:g} s -> DetER {d:.2f} (miss {miss:.2f}, FA {fa:.2f}), "
            "labels "
            f"{ev['hyp_h']:.2f} h; onset delta p10/50/90 ms {on}; offset delta {off}; "
            f"reference speech with no label {unc:.2f} %"
            + (f"; grid widened once for {', '.join(widened)}" if widened else "")
            + (f"; STILL AT THE EDGE for {', '.join(on_edge)}" if on_edge else "")
        )
        print(f"  {name}: pauses at the best setting: {pause_stats(hyp)}")
        if a.curve_pads:
            cs, ce = a.curve_pads
            cells = []
            for mp in a.min_pauses:
                ev = scorer(labels(rows, mp, cs, ce))
                cells.append(
                    f"{mp:g} s: {ev['deter']:.2f} ({ev['miss']:.2f}/{ev['fa']:.2f})"
                )
            print(
                f"  {name}: min_pause curve at {cs * 1000:.0f}/{ce * 1000:.0f} ms, "
                "DetER "
                "(miss/FA): " + "; ".join(cells)
            )
        print(
            f"  {name}: next best: "
            + "; ".join(
                f"{r[0]:.2f} ({r[3] * 1000:.0f}/{r[4] * 1000:.0f} ms, {r[5]:g} s)"
                for r in res[1:6]
            ),
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    al = sub.add_parser("align")
    al.add_argument("--set", choices=("ami", "icsi", "librispeech"), required=True)
    al.add_argument("--channel", default="", help="ami/icsi: the channel")
    al.add_argument(
        "--dump",
        required=True,
        help="ami/icsi: the test set dump; librispeech: the LibriSpeech directory",
    )
    al.add_argument("--subsets", nargs="+", default=["dev-clean", "dev-other"])
    al.add_argument(
        "--bridge",
        type=float,
        default=0.2,
        help="librispeech: MFA word gaps shorter than this are speech in mfa.rttm",
    )
    al.add_argument("--annotations", required=True)
    al.add_argument("--out", required=True)
    al.add_argument(
        "--v2",
        default=str(Path(__file__).resolve().parent),
        help="the recipe's local/v2 (default: this file's directory)",
    )
    al.add_argument("--model_tag", default="espnet/owsm_ctc_v4_1B")
    al.add_argument("--revision", default="6075bf3d42e6037d1ef7abd7c5558ade886e6eaa")
    al.add_argument("--batch_seconds", type=float, default=3840.0)
    al.add_argument("--dtype", default="float32")
    al.add_argument("--use_sdpa", type=lambda x: x.lower() == "true", default=True)
    al.add_argument("--tf32", type=lambda x: x.lower() == "true", default=False)
    al.add_argument("--min_word_score", type=float, default=-15.0)
    al.add_argument("--workers", type=int, default=12)
    al.add_argument("--utt_gap", type=float, default=1.0)
    al.add_argument("--window", type=float, default=1.0)
    al.add_argument("--bench_longest", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--out", required=True, help="one align output directory")
    sc.add_argument("--ref", required=True)
    sc.add_argument(
        "--label_set",
        choices=("segmentation", "tokens", "forced_align", "xml_words"),
        help="score only this label set (default: all in --out)",
    )
    sc.add_argument("--starts", type=float, nargs="+", default=[0, 0.025, 0.05, 0.1])
    sc.add_argument(
        "--ends", type=float, nargs="+", default=[0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    )
    sc.add_argument(
        "--min_pauses", type=float, nargs="+", default=[0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    )
    sc.add_argument(
        "--thresholds", type=float, nargs="+", default=[-2, -5, -10, -15, -20, -30]
    )
    sc.add_argument("--print_grid", action="store_true", help="every grid cell")
    sc.add_argument("--no_widen", action="store_true", help="keep the grid as given")
    sc.add_argument(
        "--curve_pads",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="also print DetER over --min_pauses at these start and end pads",
    )
    pa = sub.add_parser("pauses")
    pa.add_argument("--out", required=True)
    pa.add_argument("--ref", required=True)
    pa.add_argument("--label_set", default="tokens")
    pa.add_argument("--pad_start", type=float, default=0.05)
    pa.add_argument("--pad_end", type=float, default=0.4)
    pa.add_argument("--min_pause", type=float, default=5.0)
    pa.add_argument(
        "--inf_sets",
        nargs="+",
        default=["segmentation", "tokens", "forced_align", "xml_words"],
    )
    pa.add_argument("--starts", type=float, nargs="+", default=[0, 0.025, 0.05, 0.1])
    pa.add_argument(
        "--ends", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    a = ap.parse_args()
    {"align": cmd_align, "score": cmd_score, "pauses": cmd_pauses}[a.cmd](a)


if __name__ == "__main__":
    main()
