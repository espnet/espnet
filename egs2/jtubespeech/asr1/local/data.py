

import argparse
import re
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description="Parse JtubeSpeech dataset")

    parser.add_argument("--input_dir", type=Path, help="Original JtubeSpeech data")
    parser.add_argument("--output_dir", type=Path, help="Output data path")
    parser.add_argument("--ref_dir", type=Path, help="folder for reference files")


    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # (1) find all valid text-audio pairs
    valid_pairs = []
    for txt_file in args.input_dir.rglob("*.txt"):
        wav_file = str(txt_file).replace(".txt", ".wav").replace("/txt/", "/wav16k/")
        if Path(wav_file).exists():
            valid_pairs.append((txt_file, Path(wav_file)))
        else:
            print(f"cannot find the corrsponding wav file for {txt_file}")

    # (2) parse the list of all dev and test sets; create the writers
    # dev and eval sets are included in the test set already
    list_names = [
        # "dev_normal_jun21.list",
        # "dev_easy_jun21.list", 
        # "eval_easy_jun21.list", 
        # "eval_normal_jun21.list", 
        "test_normal_jun21.list",
    ]
    data_dict = dict(train=dict())
    for list_name in list_names:
        data_dict[list_name.replace(".list", "")] = {line.strip(): None for line in open(args.ref_dir / list_name)}
    
    for name, sub_dict in data_dict.items():
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
        sub_dict['wav.scp'] = open(args.output_dir / name / 'wav.scp', 'w')
        sub_dict['segments'] = open(args.output_dir / name / 'segments', 'w')
        sub_dict['text'] = open(args.output_dir / name / 'text', 'w')
        sub_dict['utt2spk'] = open(args.output_dir / name / 'utt2spk', 'w')
        sub_dict['spk2utt'] = open(args.output_dir / name / 'spk2utt', 'w')
    
    # (3) write all files
    for txt_file, wav_file in valid_pairs:
        stem = txt_file.stem

        for idx, line in enumerate(open(txt_file)):
            example_id = f"{stem}_{idx:04}"
            start, end, trans = line.strip().split(maxsplit=2)
            start, end = float(start), float(end)
            trans = text_norm(trans)

            partition = "train"
            for name in data_dict.keys():
                if example_id in data_dict[name]:
                    partition = name
            
            data_dict[partition]['wav.scp'].write(f"{stem} {str(wav_file)}\n")
            data_dict[partition]['segments'].write(f"{example_id} {stem} {start} {end}\n")
            data_dict[partition]['text'].write(f"{example_id} {trans}\n")
            data_dict[partition]['utt2spk'].write(f"{example_id} {example_id}\n")
            data_dict[partition]['spk2utt'].write(f"{example_id} {example_id}\n")

def text_norm(text):
    text = re.sub(r'<.*?>', '', text).strip('"').replace("#0", "")
    return text

if __name__ == "__main__":
    main()