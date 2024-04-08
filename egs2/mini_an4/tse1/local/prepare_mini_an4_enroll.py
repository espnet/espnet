import json
import random
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def prepare_mini_an4_enroll(
    wav_scp, spk2utts, output_dir, train=True, prefix="enroll_spk"
):
    uids = []
    with Path(wav_scp).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            uid = line.strip().split(maxsplit=1)[0]
            uids.append(uid)

    with Path(spk2utts).open("r", encoding="utf-8") as f:
        # {spkID: [(uid1, path1), (uid2, path2), ...]}
        spk2utt = json.load(f)

    with DatadirWriter(Path(output_dir)) as writer:
        for uid in uids:
            sid = "dummy"
            if train:
                # For training, we choose the auxiliary signal on the fly.
                # Thus, here we use the pattern f"*{uttID} {spkID}" to indicate it.
                writer[f"{prefix}1.scp"][uid] = f"*{uid} {sid}"
            else:
                enrollID = random.choice(spk2utt[sid])[1]
                while enrollID == uid and len(spk2utt[sid]) > 1:
                    enrollID = random.choice(spk2utt[sid])[1]
                writer[f"{prefix}1.scp"][uid] = enrollID


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav_scp",
        type=str,
        help="Path to the wav.scp file",
    )
    parser.add_argument(
        "spk2utts",
        type=str,
        help="Path to the json file containing mapping from speaker ID to utterances",
    )
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        help="Whether is the training set or not",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory for storing output files",
    )
    parser.add_argument(
        "--outfile_prefix",
        type=str,
        default="enroll_spk",
        help="Prefix of the output files",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    prepare_mini_an4_enroll(
        args.wav_scp,
        args.spk2utts,
        args.output_dir,
        train=args.train,
        prefix=args.outfile_prefix,
    )
