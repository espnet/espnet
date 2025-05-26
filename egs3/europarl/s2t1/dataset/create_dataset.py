import argparse
import os
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, DatasetDict
import soundfile as sf
from lhotse import CutSet, MonoCut
from lhotse.audio import Recording
from lhotse.audio.source import AudioSource
from lhotse.supervision import SupervisionSegment


def infer_langs_from_path(split_dir: Path) -> Tuple[str, str]:
    # e.g., /data/it/fr/dev => ("fr", "it")
    parent1 = split_dir.parent.name
    parent2 = split_dir.parent.parent.name
    return tuple([parent1, parent2])


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").strip().splitlines()


def build_split_cutset(
    split_dir: Path,
    src_lang: str,
    tgt_lang: str,
) -> Tuple[str, CutSet]:
    seg_lst = read_lines(split_dir / "segments.lst")
    seg_src = read_lines(split_dir / f"segments.{src_lang}")
    seg_tgt = read_lines(split_dir / f"segments.{tgt_lang}")
    spk_lst = read_lines(split_dir / "speakers.lst")
    txt_src = read_lines(split_dir / f"speeches.{src_lang}")
    txt_tgt = read_lines(split_dir / f"speeches.{tgt_lang}")
    url_lst = read_lines(split_dir / "url.lst")

    assert len(seg_lst) == len(seg_src) == len(seg_tgt)

    cuts = []
    audio_path_map = {}

    for i, line in enumerate(seg_lst):
        fname, start, end = line.strip().split()
        start = float(start)
        end = float(end)
        duration = end - start
        audio_path = split_dir.parent.parent / "audios" / f"{fname}.flac"

        if fname not in audio_path_map:
            info = sf.info(str(audio_path))
            recording = Recording(
                id=fname,
                sources=[
                    AudioSource(type="file", source=str(audio_path), channels=[0])
                ],
                sampling_rate=info.samplerate,
                num_samples=info.frames,
                duration=info.duration,
                channel_ids=[0],
            )
            audio_path_map[fname] = recording
        else:
            recording = audio_path_map[fname]

        if start > recording.duration:
            print(
                f"[Skipping] segment {fname} (start={start}, end={end}) exceeds"
                "recording duration {recording.duration}"
            )
            continue

        if end > recording.duration:
            end = recording.duration

        supervision = SupervisionSegment(
            id=f"{fname}_{i}",
            recording_id=fname,
            start=start,
            duration=duration,
            channel=0,
            text=seg_src[i],
            custom={
                f"text.{tgt_lang}": seg_tgt[i],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "speaker_id": (
                    spk_lst[txt_src.index(seg_src[i])] if seg_src[i] in txt_src else ""
                ),
                "url": (
                    url_lst[txt_src.index(seg_src[i])] if seg_src[i] in txt_src else ""
                ),
            },
        )

        cut = MonoCut(
            id=f"{fname}_cut_{i}",
            start=start,
            duration=duration,
            channel=0,
            supervisions=[supervision],
            recording=recording,
        )
        cuts.append(cut)

    split_name = (
        f"{split_dir.parent.parent.name}_{split_dir.parent.name}_{split_dir.name}"
    )
    return split_name, CutSet.from_cuts(cuts)


def convert_all(root_dir: Path, save_path: Path):
    for lang1 in os.listdir(root_dir):
        lang1_path = root_dir / lang1
        if not lang1_path.is_dir():
            continue
        for lang2 in os.listdir(root_dir / lang1):
            if lang1 == lang2:
                continue
            lang2_path = root_dir / lang1 / lang2
            if not lang2_path.is_dir():
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
                    split_name, cutset = build_split_cutset(
                        split_dir, src_lang, tgt_lang
                    )
                    out_path = save_path / f"{split_name}.jsonl.gz"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cutset.to_file(str(out_path))
                    print(f"[Saved] {split_name} with {len(cutset)} cuts to {out_path}")
                except Exception as e:
                    print(f"[Warning] Failed to process {split_dir}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=Path, help="Europarl data root (e.g. .../downloads/v1.1/)"
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Output directory to save HuggingFace dataset"
    )
    args = parser.parse_args()

    convert_all(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
