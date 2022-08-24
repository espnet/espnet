import argparse
import codecs
import os
import sys
import traceback


def get_parser():
    parser = argparse.ArgumentParser(description="prep data for lang id scoring")
    parser.add_argument(
        "--decode_folder",
        type=str,
        help="folder containing decoded text",
        required=True,
    )
    parser.add_argument(
        "--exp_folder", type=str, help="folder of experiment", required=True
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The scoring output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


def main(args):
    args = get_parser().parse_args(args)
    scoring(args.exp_folder, args.decode_folder, args.out)


def scoring(exp_folder, decode_folder, out):
    exp_decode_folder = f"{exp_folder}/{decode_folder}"
    subfolders = next(os.walk(exp_decode_folder))[1]
    for folder in subfolders:
        decode_file_name = f"{exp_decode_folder}/{folder}/text"
        decode_file = None
        output_file = open(
            f"{exp_decode_folder}/{folder}/lang_id_refs.tsv", "w", encoding="utf-8"
        )
        output_file.write(f"utt_id\tref_lid\thyp_lid\n")

        try:
            decode_file = codecs.open(decode_file_name, "r", encoding="utf-8")
        except Exception:
            traceback.print_exc()
            print("Unable to open output file: " + decode_file_name)
            continue

        utt_num = 0
        correct = 0

        while True:
            hyp = decode_file.readline()
            if not hyp:
                break

            hyp = hyp.strip().split()

            utt_id = hyp[0]
            hyp_lid = hyp[1]

            # splice out the lang id label from the utt id
            # fleurs utt id shape: {id}-{dir}-{lang_id}-audio-{split}-{audio_id}
            ref_lid = utt_id.strip().split("-")[-4]
            ref_lid = f"[{ref_lid.upper()}]"

            if ref_lid == hyp_lid:
                correct += 1
            utt_num += 1

            output_file.write(f"{utt_id}\t{ref_lid}\t{hyp_lid}\n")

        out.write(
            "Language Identification Scoring: Accuracy {:.4f} ({}/{})\n".format(
                (correct / float(utt_num)), correct, utt_num
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])
