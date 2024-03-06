import argparse
import json
import os
import sys

import pandas

dir_dict = {
    "train": "fine-tune",
    "valid": "dev",
    "test": "test",
}


def process_data(source_dir, use_classifier=False):

    for x in dir_dict:
        with open(os.path.join("data", x, "text"), "w") as text_f, open(
            os.path.join("data", x, "wav.scp"), "w"
        ) as wav_scp_f, open(os.path.join("data", x, "utt2spk"), "w") as utt2spk_f:
            metadata_f = pandas.read_csv(
                os.path.join(source_dir, "slue-hvb_" + dir_dict[x] + ".tsv"), sep="\t"
            )
            columns = list(metadata_f.columns)
            for index in range(len(metadata_f["text"])):
                if use_classifier:
                    transcript = " ".join(
                        sorted(eval(metadata_f["dialog_acts"][index]))
                    )
                else:
                    transcript = (
                        " <sep> ".join(sorted(eval(metadata_f["dialog_acts"][index])))
                        + " <utt> "
                        + metadata_f["text"][index]
                    )
                utt_id = "{}_{}_{}".format(
                    metadata_f["issue_id"][index],
                    metadata_f["start_ms"][index],
                    metadata_f["start_ms"][index] + metadata_f["duration_ms"][index],
                )
                wav_file = os.path.join(source_dir, dir_dict[x], utt_id + ".wav")
                spk_id = str(metadata_f["speaker_id"][index]) + "_spkid"
                utt_id = spk_id + "_" + utt_id

                wav_scp_f.write("{} {}\n".format(utt_id, wav_file))
                utt2spk_f.write("{} {}\n".format(utt_id, spk_id))
                text_f.write("{} {}\n".format(utt_id, transcript))


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, required=True, help="Path to source data")
parser.add_argument(
    "--use_classifier",
    type=str,
    required=False,
    default="false",
    help="Setup where prediction head are just classifier layers",
)

args = parser.parse_args()
if args.use_classifier.lower() == "false":
    use_classifier = False
else:
    use_classifier = True
process_data(args.source_dir, use_classifier)
