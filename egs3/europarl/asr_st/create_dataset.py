import argparse
import os
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, DatasetDict


def infer_langs_from_path(split_dir: Path) -> Tuple[str, str]:
    # e.g., /data/it/fr/dev => ("fr", "it")
    parent1 = split_dir.parent.name
    parent2 = split_dir.parent.parent.name
    return tuple([parent1, parent2])


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").strip().splitlines()


def build_split_dataset(
    split_dir: Path,
    src_lang: str,
    tgt_lang: str,
) -> Tuple[str, Dataset]:
    seg_lst = read_lines(split_dir / "segments.lst")
    seg_src = read_lines(split_dir / f"segments.{src_lang}")
    seg_tgt = read_lines(split_dir / f"segments.{tgt_lang}")
    spk_lst = read_lines(split_dir / "speakers.lst")
    txt_src = read_lines(split_dir / f"speeches.{src_lang}")
    txt_tgt = read_lines(split_dir / f"speeches.{tgt_lang}")
    url_lst = read_lines(split_dir / "url.lst")

    assert len(seg_lst) == len(seg_src) == len(seg_tgt)

    data = {
        "audio": [],
        f"text.{src_lang}": [],
        f"text.{tgt_lang}": [],
        "url": [],
        "speaker_id": [],
        "src_lang": [],
        "tgt_lang": [],
    }

    for i, line in enumerate(seg_lst):
        fname, start, end = line.strip().split()
        path = split_dir.parent.parent / "audios" / f"{fname}.m4a"
        data["audio"].append({"path": str(path)})
        data[f"text.{src_lang}"].append(seg_src[i])
        data[f"text.{tgt_lang}"].append(seg_tgt[i])
        data["src_lang"].append(src_lang)
        data["tgt_lang"].append(tgt_lang)
        speech_idx = txt_src.index(seg_src[i]) if seg_src[i] in txt_src else None
        data["url"].append(url_lst[speech_idx] if speech_idx is not None else "")
        data["speaker_id"].append(spk_lst[speech_idx] if speech_idx is not None else "")

    split_name = (
        f"{split_dir.parent.parent.name}_{split_dir.parent.name}_{split_dir.name}"
    )
    return split_name, Dataset.from_dict(data)


def convert_all(root_dir: Path, save_path: Path):
    dataset_dict = DatasetDict()

    for lang1 in os.listdir(root_dir):
        for lang2 in os.listdir(root_dir / lang1):
            if not (root_dir / lang1 / lang2).is_dir():
                continue

            for split in os.listdir(root_dir / lang1 / lang2):
                split_dir = root_dir / lang1 / lang2 / split
                if not split_dir.is_dir():
                    continue

                src_lang = lang1
                tgt_lang = lang2

                seg_src = split_dir / f"segments.{src_lang}"
                seg_tgt = split_dir / f"segments.{tgt_lang}"
                txt_src = split_dir / f"speeches.{src_lang}"
                txt_tgt = split_dir / f"speeches.{tgt_lang}"

                if not (
                    seg_src.exists()
                    and seg_tgt.exists()
                    and txt_src.exists()
                    and txt_tgt.exists()
                ):
                    print(f"[Skipping] Missing files in {split_dir}")
                    continue

                try:
                    split_name, dataset = build_split_dataset(
                        split_dir, src_lang, tgt_lang
                    )
                    dataset_dict[split_name] = dataset
                    print(f"[Added] {split_name} with {len(dataset)} examples")
                except Exception as e:
                    print(f"[Warning] Failed to process {split_dir}: {e}")

    save_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(save_path))
    print(f"[Saved] DatasetDict written to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=Path, help="Europarl data root (e.g. .../downloads/v1.1/)"
    )
    parser.add_argument(
        "save_dir", type=Path, help="Output directory to save HuggingFace dataset"
    )
    args = parser.parse_args()

    convert_all(args.data_dir, args.save_dir)


if __name__ == "__main__":
    main()
