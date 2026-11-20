"""Write one corpus as a VAD dump directory, read by diar.sh from stage 4.

The audio goes straight from the corpus archives into an Omni-IO archive at
16 kHz, so nothing is unpacked into the corpora directory. A dump directory has

    wav.scp           <rec-id> <Omni-IO entry>
    utt2num_samples   <rec-id> <samples>
    espnet_rttm       speech of each recording as one speaker (samples), and an
                      END line for every recording, also those without speech
    utt2category      <rec-id> <domain>, for --chunk_upsampling_factor
    manifest.tsv      one row per recording: corpus, domain, label source,
                      negative type, microphone, language, licence, source
    feats_type        raw
    text, utt2lang    (label source "A" only) the transcript and the OWSM
                      language token of each recording, for align.py, which
                      writes the speech into espnet_rttm and marks the
                      directory with .aligned

Recordings without speech (non-speech corpora) have only their END line; the
diar task gives them all-zero labels (DiarCollateFn).
"""

import json
import logging
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import resample_poly

from espnet2.fileio.omniio_audio import OmniIOAudioWriter

Interval = Tuple[float, float]

MANIFEST_FIELDS = (
    "rec_id",
    "corpus",
    "domain",
    "label_source",
    "negative_type",
    "mic",
    "lang",
    "licence",
    "source",
    "seconds",
    "speech_seconds",
    "group",
)


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Union of intervals, sorted; touching ones are merged, empty ones dropped."""
    merged: List[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def resample(signal: np.ndarray, rate: int, fs: int) -> np.ndarray:
    """`signal` (samples first) at `fs` Hz, by polyphase filtering."""
    if rate == fs:
        return signal
    g = gcd(rate, fs)
    return resample_poly(signal, fs // g, rate // g, axis=0).astype(np.float32)


class VadDumpWriter:
    """Collects the recordings of one corpus (or one shard of it).

    Args:
        out_dir: The dump directory.
        corpus: Corpus name, the prefix of every recording id.
        domain: Domain of the corpus (the category for upsampling).
        label_source: How the labels were made: "H" human segments, "D" known
            exactly (non-speech clip, or a mixture of known sources), "A"
            forced alignment of human transcripts.
        licence: Licence of the audio, for the manifest.
        fs: Sampling rate of the dump.
        max_bin_size: Maximum size of one Omni-IO bin file in bytes.
        pack_seconds: Recordings shorter than this are concatenated, in the
            order they are added, into recordings of at least this length
            (per language token and microphone), with their labels and
            transcripts; the training chunks are fixed-length windows, which a
            shorter recording cannot give. The id of a pack is "<first id>+<n>".
        max_hours: Recordings are not added once this many hours are added.
        max_hours_per_lang: The same per language (lang_sym, else lang).
    """

    def __init__(
        self,
        out_dir: Union[Path, str],
        corpus: str,
        domain: str,
        label_source: str,
        licence: str,
        fs: int = 16000,
        max_bin_size: int = 2 * 1024**3,
        pack_seconds: Optional[float] = None,
        max_hours: Optional[float] = None,
        max_hours_per_lang: Optional[float] = None,
    ):
        if label_source not in ("H", "D", "A"):
            raise ValueError(f"unknown label source: {label_source}")
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.corpus, self.domain = corpus, domain
        self.label_source, self.licence, self.fs = label_source, licence, fs
        self.audio = OmniIOAudioWriter(
            self.dir / "audio", self.dir / "wav.scp", max_bin_size=max_bin_size
        )
        self.num_samples: Dict[str, int] = {}
        self.rttm: List[str] = []
        self.rows: List[Dict[str, str]] = []
        self.skipped: List[str] = []
        self.pack_seconds = pack_seconds
        self.max_seconds = None if max_hours is None else max_hours * 3600
        self.max_seconds_lang = (
            None if max_hours_per_lang is None else max_hours_per_lang * 3600
        )
        self.before_add = None  # called at the start of every add() (prep.py --limit)
        self.pack_by_group = False  # packs hold one group each (adapters.PACK_BY_GROUP)
        self.seconds_added = 0.0
        self.seconds_lang: Dict[str, float] = {}
        self.capped = 0
        self._packs: Dict[tuple, list] = {}
        self._buffered = set()
        self.text: Dict[str, str] = {}
        self.lang_sym: Dict[str, str] = {}
        # label source A: the transcript clips of each recording, (clip id, first
        # sample, end sample, text); a pack lists its clips at their offsets
        self.clips: Dict[str, List[tuple]] = {}

    def skip(self, source: str, reason: str):
        """Record a source that could not be read (listed in skipped.txt)."""
        logging.warning("skipped %s: %s", source, reason)
        self.skipped.append(f"{source}\t{reason}")

    def add(
        self,
        rec_id: str,
        signal: np.ndarray,
        rate: int,
        speech: Sequence[Interval] = (),
        channels: Optional[Sequence[int]] = None,
        negative_type: str = "",
        mic: str = "",
        lang: str = "",
        source: str = "",
        text: Optional[str] = None,
        lang_sym: Optional[str] = None,
        group: str = "",
    ) -> int:
        """Add a recording; return the number of recordings written.

        Args:
            rec_id: Recording id within the corpus (the corpus name is
                prepended). A multi-channel signal gives one recording per
                channel, "<rec_id>_ch<k>", all with the same labels.
            signal: Samples first, (samples,) or (samples, channels).
            rate: Sampling rate of `signal` in Hz.
            speech: Speech intervals in seconds; empty for non-speech audio.
            channels: Channels to keep (default all).
            negative_type: What the non-speech is (e.g. "music", "keyboard").
            mic: "close", "far", "device", ... for the manifest.
            lang: Language code, for the manifest.
            source: Where the audio came from (archive member), for the manifest.
            text: Transcript (label source "A": align.py finds the speech).
            lang_sym: OWSM language token of the transcript, e.g. "<eng>".
            group: The group of a recording whose split is decided by group
                (registry.split_by_hash); empty for an official split.
        """
        if self.label_source == "A":
            if not text or not lang_sym or speech:
                raise ValueError(
                    f"{rec_id}: label source A takes text and lang_sym, not speech"
                )
        if self.before_add is not None:
            self.before_add()
        signal = np.asarray(signal, dtype=np.float32)
        key = lang_sym or lang
        seconds = len(signal) / rate
        if (
            self.max_seconds is not None and self.seconds_added >= self.max_seconds
        ) or (
            self.max_seconds_lang is not None
            and self.seconds_lang.get(key, 0.0) >= self.max_seconds_lang
        ):
            self.capped += 1
            return 0
        self.seconds_added += seconds
        self.seconds_lang[key] = self.seconds_lang.get(key, 0.0) + seconds
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.ndim != 2 or signal.shape[0] == 0:
            raise ValueError(f"{rec_id}: bad signal shape {signal.shape}")
        signal = resample(signal, rate, self.fs)
        keep = range(signal.shape[1]) if channels is None else channels
        multi = signal.shape[1] > 1 and (channels is None or len(channels) > 1)
        intervals = merge_intervals(speech)
        written = 0
        for k in keep:
            rid = f"{self.corpus}-{rec_id}" + (f"_ch{k}" if multi else "")
            if rid in self.num_samples or rid in self._buffered:
                raise ValueError(f"duplicate recording id {rid}")
            x = np.clip(signal[:, k], -1.0, 1.0)
            meta = dict(
                negative_type=negative_type,
                mic=mic,
                lang=lang,
                source=source,
                group=group,
            )
            if self.pack_seconds and len(x) < self.pack_seconds * self.fs:
                written += self._buffer(rid, x, intervals, meta, text, lang_sym)
            else:
                self._write(rid, x, intervals, meta, text, lang_sym)
                written += 1
        return written

    def _buffer(self, rid, x, intervals, meta, text, lang_sym) -> int:
        """Queue a short recording for packing; write a pack once it is long enough.

        Recordings are packed in the order they come, separately per language
        token and microphone, so a pack has one language for the alignment.
        """
        key = (lang_sym, meta["mic"])
        if self.pack_by_group:
            key += (meta["group"],)
        pack = self._packs.setdefault(key, [])
        pack.append((rid, x, intervals, meta, text))
        self._buffered.add(rid)
        if sum(len(p[1]) for p in pack) >= self.pack_seconds * self.fs:
            self._flush(key)
            return 1
        return 0

    def _flush(self, key):
        """Write the queued recordings of `key` as one recording."""
        pack = self._packs.pop(key, [])
        if not pack:
            return
        offset, xs, intervals, texts, sources, clips = 0, [], [], [], [], []
        for rid, x, ivs, meta, text in pack:
            t0 = offset / self.fs
            intervals += [(t0 + s, t0 + e) for s, e in ivs]
            xs.append(x)
            if text is not None:
                texts.append(text)
                clips.append((rid, offset, offset + len(x), " ".join(text.split())))
            offset += len(x)
            sources.append(meta["source"] or rid)
        first = pack[0]
        meta = dict(first[3])
        meta["source"] = ";".join(sources)
        types = sorted({p[3]["negative_type"] for p in pack if p[3]["negative_type"]})
        meta["negative_type"] = ",".join(types)
        # every group of a pack: splits.py keeps a pack whole in one split
        meta["group"] = ";".join(sorted({p[3]["group"] for p in pack if p[3]["group"]}))
        text = " ".join(texts) if texts else None
        self._write(
            f"{first[0]}+{len(pack)}",
            np.concatenate(xs),
            intervals,
            meta,
            text,
            key[0],
            clips=clips,
        )

    def _write(self, rid, x, intervals, meta, text, lang_sym, clips=None):
        """Write one mono recording with its speech intervals (seconds).

        clips: the transcript clips of a pack (_flush); a recording with a
        transcript that is not a pack is one clip.
        """
        n = len(x)
        if text is not None:
            self.clips[rid] = clips or [(rid, 0, n, " ".join(text.split()))]
        self.audio[rid] = self.fs, x
        self.num_samples[rid] = n
        speech_samples = 0
        for start, end in intervals:
            s = max(0, int(round(start * self.fs)))
            e = min(n, int(round(end * self.fs)))
            if e > s:
                # espnet RTTM: start and end in samples (convert_rttm.py)
                self.rttm.append(f"SPEAKER {rid} 1 {s} {e} <NA> <NA> {rid} <NA>\n")
                speech_samples += e - s
        self.rttm.append(f"END {rid} <NA> <NA> {n} <NA> <NA> <NA> <NA>\n")
        if text is not None:
            self.text[rid] = " ".join(text.split())
            self.lang_sym[rid] = lang_sym
        self.rows.append(
            dict(
                rec_id=rid,
                corpus=self.corpus,
                domain=self.domain,
                label_source=self.label_source,
                licence=self.licence,
                seconds=f"{n / self.fs:.3f}",
                speech_seconds=f"{speech_samples / self.fs:.3f}",
                **meta,
            )
        )

    def close(self):
        """Write the remaining packs and the text files."""
        for key in list(self._packs):
            self._flush(key)
        skipped = self.skipped
        self.audio.close()
        ids = sorted(self.num_samples)
        (self.dir / "utt2num_samples").write_text(
            "".join(f"{r} {self.num_samples[r]}\n" for r in ids)
        )
        (self.dir / "utt2category").write_text(
            "".join(f"{r} {self.domain}\n" for r in ids)
        )
        (self.dir / "espnet_rttm").write_text("".join(sorted(self.rttm)))
        (self.dir / "feats_type").write_text("raw\n")
        if self.label_source == "A":
            (self.dir / "text").write_text(
                "".join(f"{r} {self.text[r]}\n" for r in ids)
            )
            (self.dir / "utt2lang").write_text(
                "".join(f"{r} {self.lang_sym[r]}\n" for r in ids)
            )
            # <rec_id> <clip id> <first sample> <end sample> <text>, in order
            (self.dir / "clips").write_text(
                "".join(
                    f"{r}\t{c}\t{s}\t{e}\t{t}\n"
                    for r in ids
                    for c, s, e, t in self.clips[r]
                )
            )
        with (self.dir / "manifest.tsv").open("w") as f:
            f.write("\t".join(MANIFEST_FIELDS) + "\n")
            for row in sorted(self.rows, key=lambda r: r["rec_id"]):
                f.write("\t".join(row[k] for k in MANIFEST_FIELDS) + "\n")
        (self.dir / "skipped.txt").write_text("".join(f"{s}\n" for s in skipped))
        total = sum(float(r["seconds"]) for r in self.rows)
        speech = sum(float(r["speech_seconds"]) for r in self.rows)
        summary = dict(
            corpus=self.corpus,
            domain=self.domain,
            recordings=len(ids),
            hours=round(total / 3600, 3),
            speech_hours=round(speech / 3600, 3),
            skipped=len(skipped),
            not_added_over_cap=self.capped,
        )
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        logging.info("%s", summary)
        if skipped:
            logging.warning("%d unreadable sources in %s", len(skipped), self.dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
