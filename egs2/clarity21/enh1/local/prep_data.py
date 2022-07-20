import argparse
import json
import os

parser = argparse.ArgumentParser("Clarity")
parser.add_argument(
    "--clarity_root",
    type=str,
    help="Path to Clarity Challenge root folder "
    "(Folder containing train, dev and metadata dirs)",
)
parser.add_argument(
    "--fs",
    type=int,
    default=16000,
    help="Sample rate to use, by default we resample to 16000 Hz",
)


def prepare_data(clarity_root, samplerate):

    output_folder = "./data"
    ids = {"train": set(), "dev": set()}

    for ds_split in ids.keys():
        metafile = os.path.join(
            clarity_root, "metadata", "scenes.{}.json".format(ds_split)
        )
        with open(metafile, "r") as f:
            metadata = json.load(f)
        for ex in metadata:
            ids[ds_split].add(ex["scene"])

    for ds_split in ids.keys():
        ids[ds_split] = sorted(list(ids[ds_split]))

    # create wav.scp
    for ds_split in ids.keys():
        os.makedirs(os.path.join(output_folder, ds_split), exist_ok=True)
        with open(os.path.join(output_folder, ds_split, "wav.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_files = [
                    os.path.join(
                        clarity_root,
                        ds_split,
                        "scenes",
                        "{}_mixed_CH{}.wav".format(ex_id, idx),
                    )
                    for idx in range(1, 4)
                ]

                assert all([os.path.exists(x) for x in array_files]), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                array_files = " ".join(array_files)
                f.write(
                    "{} sox -M {} -c 6 -b 16 -r {} -t wav - |\n".format(
                        ex_id, array_files, samplerate
                    )
                )

        with open(os.path.join(output_folder, ds_split, "noise1.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_file = os.path.join(
                    clarity_root,
                    ds_split,
                    "scenes",
                    "{}_interferer_CH1.wav".format(ex_id),
                )
                assert os.path.exists(array_file), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                f.write(
                    "{} sox {} -b 16 -r {} -t wav - remix 1 |\n".format(
                        ex_id, array_file, samplerate
                    )
                )

        with open(os.path.join(output_folder, ds_split, "spk1.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_file = os.path.join(
                    clarity_root, ds_split, "scenes", "{}_target_CH1.wav".format(ex_id)
                )
                assert os.path.exists(array_file), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                f.write(
                    "{} sox {}  -b 16 -r {} -t wav - remix 1 |\n".format(
                        ex_id, array_file, samplerate
                    )
                )

        with open(os.path.join(output_folder, ds_split, "noise.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_file = os.path.join(
                    clarity_root,
                    ds_split,
                    "scenes",
                    "{}_interferer_CH1.wav".format(ex_id),
                )
                assert os.path.exists(array_file), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                f.write(
                    "{} sox {}  -b 16 -r {} -t wav - remix 1 |\n".format(
                        ex_id, array_file, samplerate
                    )
                )

        with open(os.path.join(output_folder, ds_split, "text.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                f.write("{} dummy\n".format(ex_id))

        with open(os.path.join(output_folder, ds_split, "utt2spk"), "w") as f:
            for ex_id in ids[ds_split]:
                f.write("{} dummy\n".format(ex_id))

        with open(os.path.join(output_folder, ds_split, "spk2utt"), "w") as f:
            for ex_id in ids[ds_split]:
                f.write("dummy {}\n".format(ex_id))


if __name__ == "__main__":
    args = parser.parse_args()
    prepare_data(args.clarity_root, args.fs)
