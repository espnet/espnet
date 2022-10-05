import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--wavscp", type=str, required=True)
    parser.add_argument("--reco-list", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=30)
    return parser.parse_args()


def main():
    args = get_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_wav_scp = open(os.path.join(args.out_dir, "wav.scp"), "w", encoding="utf-8")
    out_segments = open(os.path.join(args.out_dir, "segments"), "w", encoding="utf-8")
    too_short_path = os.path.join(args.out_dir, "too_short.list")
    too_short = open(too_short_path, "w", encoding="utf-8")
    too_long_path = os.path.join(args.out_dir, "too_long.list")
    too_long = open(too_long_path, "w", encoding="utf-8")

    reco2path = {}
    with open(args.wavscp) as f:
        for line in f:
            reco, path = line.rstrip("\n").split()
            reco2path[reco] = path

    lengths = []
    n_too_short = 0
    n_too_long = 0
    with open(args.reco_list) as f:
        for line in f:
            seg = line.rstrip("\n")
            reco, timestamps = seg.split("-")  # utt = f'{reco}-{start_time}_{end_time}'
            start, end = timestamps.split("_")

            start = float(start) / 1000
            end = float(end) / 1000  # ms to seconds
            if end - start < args.min_duration:
                # print(f"WARNING: {seg} too short (<{args.min_duration}s)")
                n_too_short += 1
                too_short.write(f"{seg}\t{end - start}\n")
                continue

            # TODO: manually cut these audio files
            if end - start > args.max_duration:
                # print(f"WARNING: {seg} too long (>{args.max_duration}s)")
                n_too_long += 1
                too_long.write(f"{seg}\t{end - start}\n")
                continue

            lengths.append(end - start)

            if reco not in reco2path:
                # print(f'WARNING: cannot find {reco} in {args.wavscp}')
                continue

            wav_path = reco2path[reco]

            if not os.path.exists(wav_path):
                print(f"WARNING: {wav_path} does not exist")
                continue

            out_wav_scp.write(f"{reco}\t{wav_path}\n")
            out_segments.write(f"{seg}\t{reco}\t{start}\t{end}\n")

    print(
        f"WARNING: {n_too_short} utterances are too short (<{args.min_duration}s), see {too_short_path}"
    )
    print(
        f"WARNING: {n_too_short} utterances are too long (>{args.max_duration}s), see {too_long_path}"
    )


if __name__ == "__main__":
    main()
