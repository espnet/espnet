import argparse
import re
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description="Parse LaboroTVSpeech dataset")

    parser.add_argument("--input_dir", type=Path, help="Original LaboroTVSpeech data")
    parser.add_argument("--output_dir", type=Path, help="Output data path")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for partition in ["train", "dev"]:
        data_dict = dict()

        (args.output_dir / partition).mkdir(parents=True, exist_ok=True)
        wav_scp_writer = open(args.output_dir / partition / 'wav.scp', 'w')
        text_writer = open(args.output_dir / partition / 'text', 'w')
        utt2spk_writer = open(args.output_dir / partition / 'utt2spk', 'w')
        spk2utt_writer = open(args.output_dir / partition / 'spk2utt', 'w')

        for line in open(args.input_dir / partition / "text.csv"):
            example_id, trans = line.strip().split(",", maxsplit=1)
            trans = text_norm(trans)
            data_dict[example_id] = trans

        for file in (args.input_dir / partition).rglob("*.wav"):
            example_id = file.stem

            if example_id in data_dict:
                wav_scp_writer.write(f"{example_id} {str(file)}\n")
                text_writer.write(f"{example_id} {data_dict[example_id]}\n")
                utt2spk_writer.write(f"{example_id} {example_id}\n")
                spk2utt_writer.write(f"{example_id} {example_id}\n")

def text_norm(text):
    text = text.strip().split()
    text = [t.split("+")[0] for t in text]
    text = "".join(text)
    return text

if __name__ == "__main__":
    main()