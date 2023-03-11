import yaml
import argparse

def get_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, help="Path to original corpus.")
    parser.add_argument("--out", "-o", type=str, help="Path to processed files.")
    args = parser.parse_args()

    return args

def yaml2files(data_path: str, out_path: str):
    split = data_path.split("/")[-2]
    split = "train" if split == "train_full" else split

    data = {}
    with open(f"{data_path}/txt/{split}.fra", "r", encoding="utf-8") as org_text_file, \
        open(f"{data_path}/txt/{split}.yaml", "r", encoding="utf-8") as yaml_file, \
        open(f"{out_path}/text.fr", "w", encoding="utf-8") as processed_text_file, \
        open(f"{out_path}/wav.scp", "w", encoding="utf-8") as wav_scp_file, \
        open(f"{out_path}/utt2spk", "w", encoding="utf-8") as utt2spk_file:
        text_lines = org_text_file.readlines()
        raw_yaml = yaml.safe_load(yaml_file)

        for text_line, row in zip(text_lines, raw_yaml):
            text_line = text_line.strip()
            uid = row["wav"].strip()
            speaker_id = row["speaker_id"].strip()
            
            wav_path = f"{data_path}/wav/{uid}.wav"

            processed_text_file.write(f"{uid} {text_line}\n")
            wav_scp_file.write(f"{uid} {wav_path}\n")
            utt2spk_file.write(f"{uid} {uid}-{speaker_id}\n")


if __name__ == "__main__":
    args = get_args()
    
    data_path = args.data
    out_path = args.out

    yaml2files(data_path, out_path)
