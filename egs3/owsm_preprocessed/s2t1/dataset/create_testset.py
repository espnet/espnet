from pathlib import Path
import argparse
from datasets import Dataset, DatasetDict
import soundfile as sf
from tqdm import tqdm
import numpy as np


def read_pcm(filepath, sample_rate=16000, dtype=np.int16):
    data = np.fromfile(filepath, dtype=dtype)
    return data.astype(np.float32) / np.iinfo(dtype).max


def prepare_mls(input_dir: Path, output_dir: Path) -> DatasetDict:
    dataset_dict = {}

    for lang_dir in sorted(input_dir.glob("mls_*")):
        if "lm" in lang_dir.name:
            continue  # Skip LM directories

        lang = lang_dir.name.split("_", 1)[-1]
        test_dir = lang_dir / "test"
        audio_dir = test_dir / "audio"
        transcript_path = test_dir / "transcripts.txt"
        segments_path = test_dir / "segments.txt"

        examples = {"utt_id": [], "audio": [], "text": []}

        # Build a map from utt_id to transcript
        with open(transcript_path, encoding="utf-8") as ft:
            transcript_map = {
                line.split('\t', maxsplit=1)[0]: line.split('\t', maxsplit=1)[1].strip()
                for line in ft if line.strip()
            }


        for line in tqdm(open(segments_path, encoding="utf-8"), desc=f"Processing {lang}"):
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            utt_id, _, _, _ = parts
            path_parts = list(utt_id.split("_")[:2])
            filename = f"{utt_id}.flac"
            wav_path = audio_dir.joinpath(*path_parts, filename)

            if not wav_path.exists():
                continue

            try:
                audio, _ = sf.read(wav_path)
            except:
                continue  # Skip broken files

            examples["utt_id"].append(utt_id)
            examples["audio"].append({"path": str(wav_path)})
            examples["text"].append(transcript_map.get(utt_id, ""))

        dataset = Dataset.from_dict(examples)
        dataset_dict[lang] = dataset

    return DatasetDict(dataset_dict)


def prepare_ksponspeech(input_dir: Path, output_dir: Path) -> DatasetDict:
    eval_dir = input_dir / "KsponSpeech_eval"
    examples = {"utt_id": [], "audio": [], "text": []}

    for subdir in sorted(eval_dir.glob("KsponSpeech_*")):
        for pcm_file in sorted(subdir.glob("*.pcm")):
            utt_id = pcm_file.stem
            txt_file = pcm_file.with_suffix(".txt")
            if not txt_file.exists():
                continue

            with open(txt_file, encoding="utf-8") as f:
                text = f.read().strip()

            examples["utt_id"].append(utt_id)
            examples["audio"].append({"path": str(pcm_file)})
            examples["text"].append(text)

    dataset = Dataset.from_dict(examples)
    return DatasetDict({"eval": dataset})


def prepare_covost2(input_dir: Path, output_dir: Path) -> DatasetDict:
    """Prepare and return a DatasetDict for the CoVoST2 dataset."""
    raise NotImplementedError("CoVoST2 preparation not yet implemented")


def prepare_reasonspeech(input_dir: Path, output_dir: Path) -> DatasetDict:
    """Prepare and return a DatasetDict for ReazonSpeech test data."""
    tsv_path = input_dir / "small.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"{tsv_path} not found")

    examples = {"utt_id": [], "audio": [], "text": []}

    with open(tsv_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading ReazonSpeech TSV"):
            line = line.strip()
            if not line:
                continue
            rel_path, text = line.split("\t", maxsplit=1)
            flac_path = input_dir / rel_path
            utt_id = flac_path.stem

            if not flac_path.exists():
                continue

            examples["utt_id"].append(utt_id)
            examples["audio"].append({"path": str(flac_path)})
            examples["text"].append(text)

    dataset = Dataset.from_dict(examples)
    return DatasetDict({"test": dataset})


def main(input_dir: str, output_dir: str, dataset: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "mls":
        dataset_dict = prepare_mls(input_dir, output_dir)
    elif dataset == "ksponspeech":
        dataset_dict = prepare_ksponspeech(input_dir, output_dir)
    elif dataset == "covost2":
        dataset_dict = prepare_covost2(input_dir, output_dir)
    elif dataset == "reasonspeech":
        dataset_dict = prepare_reasonspeech(input_dir, output_dir)
    else:
        raise ValueError(f"Unknown dataset name: {dataset}")

    dataset_dict.save_to_disk(str(output_dir))
    print(f"\n✅ DatasetDict for '{dataset}' saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DatasetDict for test data")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory to save DatasetDict")
    parser.add_argument("--dataset", type=str, required=True, choices=["mls", "ksponspeech", "covost2", "reasonspeech"],
                        help="Dataset name to prepare")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.dataset)
