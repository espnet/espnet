import argparse

parser = argparse.ArgumentParser(description="Download and format FLEURS dataset")
parser.add_argument(
    "--data_dir",
    default="dump/raw/train_all",
    type=str,
    help="directory containing asr data",
)

args = parser.parse_args()

with open(f"{args.data_dir}/text", "r", encoding="utf-8") as in_file, open(
    f"{args.data_dir}/lid_utt", "w", encoding="utf-8"
) as utt_file, open(f"{args.data_dir}/lid_tok", "w", encoding="utf-8") as tok_file:
    lines = in_file.readlines()
    for line in lines:
        utt_id = line.split()[0]
        lid = line[line.index("[") : line.index("]") + 1]
        utt_file.write(f"{utt_id} {lid} \n")

        words = line[line.index("]") + 1 :]
        lids = [lid for word in words.split()]
        tok_file.write(f"{utt_id} {lid} {' '.join(lids)}\n")
