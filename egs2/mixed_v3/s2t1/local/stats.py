from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def collect_stats(data_dir: Path):
    stats = defaultdict(float)
    src_langs = []
    with open(data_dir / "text", "r") as fp:
        for line in fp:
            line = line.strip().split(maxsplit=1)[-1]
            duration = float(line.split("<")[-1][:-1])
            src = line.split(">")[0][1:]
            tgt = line.split("><")[1].split("_")[-1]
            if tgt == "asr":
                tgt = src
            stats[(src, tgt)] += duration
            if src not in src_langs:
                src_langs.append(src)
    return stats, sorted(src_langs)


def get_parser():
    parser = ArgumentParser(description="Show statistics of the data directory.")
    parser.add_argument(
        "-d", "--data_dir", type=Path, required=True, help="Data directory."
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    stats, langs = collect_stats(args.data_dir)
    stats = sorted(list(stats.items()), key=lambda x: x[-1], reverse=True)
    total_duration = sum(x[1] for x in stats)
    asr_duration = sum(x[1] for x in stats if x[0][0] == x[0][1])
    st_duration = sum(x[1] for x in stats if x[0][0] != x[0][1])

    with open(args.data_dir / "stats.txt", "w") as fp:
        fp.write(f"Languages: {len(langs)}\n")
        fp.write("\n".join(langs))
        fp.write(f"\nTotal duration: {total_duration / 60 / 60:.3f} hours\n")
        fp.write(f"ASR duration: {asr_duration / 60 / 60:.3f} hours\n")
        fp.write(f"Translation duration: {st_duration / 60 / 60:.3f} hours\n")
        for (src, tgt), duration in stats:
            fp.write(f"{src} -> {tgt}: {duration / 60 / 60:.3f} hours\n")
