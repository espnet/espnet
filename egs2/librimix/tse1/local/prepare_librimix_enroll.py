import json
import random
from itertools import chain
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def prepare_librimix_enroll(
    wav_scp, spk2utts, output_dir, num_spk=2, train=True, prefix="enroll_spk"
):
    mixtures = []
    with Path(wav_scp).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mixtureID = line.strip().split(maxsplit=1)[0]
            mixtures.append(mixtureID)

    with Path(spk2utts).open("r", encoding="utf-8") as f:
        # {spkID: [(uid1, path1), (uid2, path2), ...]}
        spk2utt = json.load(f)

    with DatadirWriter(Path(output_dir)) as writer:
        for mixtureID in mixtures:
            # 100-121669-0004_3180-138043-0053
            uttIDs = mixtureID.split("_")
            for spk in range(num_spk):
                uttID = uttIDs[spk]
                spkID = uttID.split("-")[0]
                if train:
                    # For training, we choose the auxiliary signal on the fly.
                    # Thus, here we use the pattern f"*{uttID} {spkID}" to indicate it.
                    writer[f"{prefix}{spk + 1}.scp"][mixtureID] = f"*{uttID} {spkID}"
                else:
                    enrollID = random.choice(spk2utt[spkID])[1]
                    while enrollID == uttID and len(spk2utt[spkID]) > 1:
                        enrollID = random.choice(spk2utt[spkID])[1]
                    writer[f"{prefix}{spk + 1}.scp"][mixtureID] = enrollID


def prepare_librimix_enroll_v2(
    wav_scp, librimix_dir, map_mix2enroll, output_dir, num_spk=2, prefix="enroll_spk"
):
    # noqa E501: ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/create_enrollment_csv_fixed.py
    mixtures = []
    with Path(wav_scp).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mixtureID = line.strip().split(maxsplit=1)[0]
            mixtures.append(mixtureID)

    utt2path = {}
    for audio in chain(
        Path(librimix_dir).rglob("s1/*.wav"),
        Path(librimix_dir).rglob("s2/*.wav"),
        Path(librimix_dir).rglob("s3/*.wav"),
    ):
        pdir = audio.parent.stem
        utt2path[pdir + "/" + audio.stem] = str(audio.resolve())

    mix2enroll = {}
    with open(map_mix2enroll) as f:
        for line in f:
            mix_id, utt_id, enroll_id = line.strip().split()
            sid = mix_id.split("_").index(utt_id) + 1
            mix2enroll[mix_id, f"s{sid}"] = enroll_id

    with DatadirWriter(Path(output_dir)) as writer:
        for mixtureID in mixtures:
            # 100-121669-0004_3180-138043-0053
            for spk in range(num_spk):
                enroll_id = mix2enroll[mixtureID, f"s{spk + 1}"]
                writer[f"{prefix}{spk + 1}.scp"][mixtureID] = utt2path[enroll_id]


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
        "--num_spk",
        type=int,
        default=2,
        choices=(2, 3),
        help="Number of speakers in each mixture sample",
    )
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        help="Whether is the training set or not",
    )
    parser.add_argument(
        "--librimix_dir",
        type=str,
        default=None,
        help="Path to the generated LibriMix directory. "
        "If `train` is False, this value is required.",
    )
    parser.add_argument(
        "--mix2enroll",
        type=str,
        default=None,
        help="Path to the downloaded map_mixture2enrollment file. "
        "If `train` is False, this value is required.",
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

    if args.train:
        prepare_librimix_enroll(
            args.wav_scp,
            args.spk2utts,
            args.output_dir,
            num_spk=args.num_spk,
            train=args.train,
            prefix=args.outfile_prefix,
        )
    else:
        prepare_librimix_enroll_v2(
            args.wav_scp,
            args.librimix_dir,
            args.mix2enroll,
            args.output_dir,
            num_spk=args.num_spk,
            prefix=args.outfile_prefix,
        )
