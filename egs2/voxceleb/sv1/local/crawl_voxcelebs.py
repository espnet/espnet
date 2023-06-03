import argparse
import os
import subprocess
import sys


def get_stt_end_indices(indir):
    with open(indir, "r") as f:
        lines = f.readlines()

    idx = 0
    while True:
        chunks = lines[idx].split("\t")
        if len(chunks) != 5:
            idx += 1
            continue
        if lines[idx][0] == "F":
            idx += 1
            continue
        stt = float(chunks[0]) / 25  # (25fps)
        break

    return stt, float(lines[-1].split("\t")[0]) / 25  # (25fps)


def main(args):
    root_dir = args.root_dir
    assert root_dir != "", "'root_dir' should be configured!"

    for r, ds, fs in os.walk(root_dir):
        for f in fs:
            print("=" * 10)
            print(r, f)  # debug
            if os.path.splitext(f)[1] != ".txt":
                continue
            if f[0] in [".", "_"]:
                continue

            yt_link = r.strip().split("/")[-1]

            stt, end = get_stt_end_indices(os.path.join(r, f))
            out_dir = os.path.join(r, f).replace(".txt", ".wav")
            print(f"Download {yt_link} {stt} to {end}, save to {out_dir}")

            cmd = (
                f"ffmpeg -y -ss {stt} -to {end}"
                f" -i $(youtube-dl -g https://www.youtube.com/watch?v={yt_link}"
                f" -x --audio-format best --audio-quality 0) -ac 1 -ar 16000 {out_dir}"
            )
            print(cmd)  # debug
            subprocess.run([cmd], shell=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCeleb 1&2 downloader")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="root directory of voxcelebs",
    )
    args = parser.parse_args()

    sys.exit(main(args))
