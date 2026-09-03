#!/usr/bin/env python3
"""Build an espnet data dir (and its dump/raw/org twin) from a clips parquet.

The pretraining corpus ships a per-corpus/language parquet describing every
clip: `rel_path`, `measured_duration_s`, and (for some corpora) `sample_rate`,
`channels`, `codec`.  Every file is 16 kHz mono FLAC, which is exactly the
`save_asis` case in format_wav_scp.py -- stage 3 would copy the source path
into wav.scp untouched and only write utt2num_samples.  Both of those we can
get from the metadata, so this reproduces stage 3's output directly instead of
opening 62 M file headers.

utt2num_samples is consumed only by the stage-4 length filter (wavlm.sh:310);
nothing under espnet2/ reads it, so the millisecond quantisation in youtube's
`measured_duration_s` (<= 8 samples) cannot affect training.

utt_id = <corpus>-<lang>-<path under lang, '/'->'-' and '.'->'_'>, which is the
same scheme local/data_pretraining.sh used when it scanned the filesystem.

With --audio-scan, any file present on disk but absent from the parquet is still
included, with its length read from the FLAC header. common_voice/ps ships
110,738 such clips (~104 h); the parquet is not a quality filter there -- it
already carries the `invalidated` and `other` splits -- so they are kept.
"""
import argparse
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--lang", required=True)
    p.add_argument("--db-root", required=True)
    p.add_argument("--data-dir", required=True, help="data/<dset>")
    p.add_argument("--dump-dir", required=True, help="dump/raw/org/<dset>")
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--audio-ext", default="flac")
    p.add_argument(
        "--audio-scan",
        default=None,
        help="optional file listing every rel_path on disk for this corpus/lang; "
        "entries missing from the parquet are added with header-read lengths",
    )
    args = p.parse_args()

    t = pq.read_table(args.parquet, columns=["rel_path", "measured_duration_s", "lang"])

    langs = set(pc.unique(t.column("lang")).to_pylist())
    if langs != {args.lang}:
        raise SystemExit(f"{args.parquet}: expected lang {{{args.lang}}}, found {langs}")

    rel = t.column("rel_path")
    prefix = f"audio/{args.corpus}/{args.lang}/"
    if pc.sum(pc.invert(pc.starts_with(rel, prefix))).as_py():
        raise SystemExit(f"{args.parquet}: some rel_path do not start with {prefix}")

    # <rest> = path under the language dir, extension stripped
    rest = pc.replace_substring(rel, prefix, "")
    rest = pc.replace_substring(rest, f".{args.audio_ext}", "")
    ident = pc.binary_join_element_wise(
        f"{args.corpus}-{args.lang}-",
        pc.replace_substring(pc.replace_substring(rest, "/", "-"), ".", "_"),
        "",
    )
    path = pc.binary_join_element_wise(f"{args.db_root}/", rel, "")

    dur = t.column("measured_duration_s")
    if pc.sum(pc.is_null(dur)).as_py():
        raise SystemExit(f"{args.parquet}: null measured_duration_s")
    nsamp = pc.cast(pc.round(pc.multiply(pc.cast(dur, "float64"), float(args.fs))), "int64")

    ident = ident.to_pylist()
    path = path.to_pylist()
    nsamp = nsamp.to_pylist()

    if args.audio_scan:
        known = set(rel.to_pylist())
        extra = [
            l.rstrip("\n") for l in open(args.audio_scan) if l.rstrip("\n") not in known
        ]
        for r in extra:
            fp = f"{args.db_root}/{r}"
            info = sf.info(fp)
            if info.samplerate != args.fs or info.channels != 1:
                raise SystemExit(f"{fp}: {info.samplerate} Hz, {info.channels} ch")
            rest = r[len(prefix):].rsplit(f".{args.audio_ext}", 1)[0]
            ident.append(f"{args.corpus}-{args.lang}-" + rest.replace("/", "-").replace(".", "_"))
            path.append(fp)
            nsamp.append(info.frames)
        if extra:
            print(f"# {args.corpus}/{args.lang}: +{len(extra)} not in parquet", flush=True)

    order = sorted(range(len(ident)), key=ident.__getitem__)
    ident = [ident[i] for i in order]
    path = [path[i] for i in order]
    nsamp = [nsamp[i] for i in order]

    if len(set(ident)) != len(ident):
        raise SystemExit(f"{args.parquet}: duplicate utt_ids after id construction")

    for d in (args.data_dir, args.dump_dir):
        os.makedirs(d, exist_ok=True)

    # Stream every file from a generator: the largest corpus is 5 M rows and
    # several of these run in parallel, so materialising the lines would cost
    # more memory than the parquet itself.
    def dump(d, name, lines):
        with open(os.path.join(d, name), "w") as f:
            f.writelines(lines)

    for d in (args.data_dir, args.dump_dir):
        dump(d, "wav.scp", (f"{i} {p}\n" for i, p in zip(ident, path)))
        dump(d, "utt2spk", (f"{i} {i}\n" for i in ident))
        dump(d, "spk2utt", (f"{i} {i}\n" for i in ident))
        dump(d, "text", (f"{i} <dummy>\n" for i in ident))
    dump(args.dump_dir, "utt2num_samples",
         (f"{i} {n}\n" for i, n in zip(ident, nsamp)))
    with open(os.path.join(args.dump_dir, "feats_type"), "w") as f:
        f.write("raw\n")

    hours = sum(nsamp) / args.fs / 3600
    print(f"{args.corpus}/{args.lang}\t{len(ident)}\t{hours:.1f}")


if __name__ == "__main__":
    main()
