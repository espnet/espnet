from pathlib import Path
from datasets import IterableDataset, Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
import subprocess


def generate_examples_sharded(wav_scp_path, text_path, text_ctc_path, text_prev_path,
                               shard_id: int, num_shards: int, expected_len: int):
    # Generator that yields examples assigned to a specific shard
    with open(wav_scp_path, "r", encoding="utf-8") as wav_scp_reader, \
         open(text_path, "r", encoding="utf-8") as text_reader, \
         open(text_ctc_path, "r", encoding="utf-8") as text_ctc_reader, \
         open(text_prev_path, "r", encoding="utf-8") as text_prev_reader:

        for i, (wav_line, text_line, text_ctc_line, text_prev_line) in tqdm(enumerate(zip(
            wav_scp_reader, text_reader, text_ctc_reader, text_prev_reader
        )), total=expected_len, desc="Creating shards"):
            if num_shards == 1 or (i % num_shards == shard_id):
                yield {
                    "utt_id": wav_line.strip().split(maxsplit=1)[0],
                    "audio_path": wav_line.strip().split(maxsplit=1)[1],
                    "text": text_line.strip().split(maxsplit=1)[1],
                    "text_ctc": text_ctc_line.strip().split(maxsplit=1)[1],
                    "text_prev": text_prev_line.strip().split(maxsplit=1)[1],
                }

def count_lines(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    return int(result.stdout.split()[0])


def create_sharded_datasets(input_dir, output_dir, split, num_shards=1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    data_path = input_dir / "dump" / "raw" / "org" / split
    wav_scp = data_path / "wav.scp"
    text = data_path / "text"
    text_ctc = data_path / "text.ctc"
    text_prev = data_path / "text.prev"

    # Check that all files have the same number of lines
    print(f"==> Checking files...")
    expected_len = count_lines(wav_scp)
    for f in [text, text_ctc, text_prev]:
        print(f"==> Check length of {f.name}...")
        assert count_lines(f) == expected_len, "Line count mismatch among input files."

    # Iterate through all shard IDs and save each one individually
    # for shard_id in range(num_shards):
    shard_id = 11
    print(f"==> Processing shard {shard_id+1}/{num_shards}...")

    dataset_iter = IterableDataset.from_generator(
        lambda: generate_examples_sharded(
            wav_scp, text, text_ctc, text_prev,
            shard_id=shard_id, num_shards=num_shards,
            expected_len=expected_len,
        )
    )

    dataset = Dataset.from_list(list(dataset_iter))

    shard_key = f"{split}/{shard_id}" if num_shards > 1 else split
    save_path = output_dir / shard_key
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))

    print(f"\n✅ Shard {shard_id} saved to {save_path}")

def load_sharded_datasetdict(output_dir, split, num_shards):
    # Load all saved shards and return a dictionary of datasets
    output_dir = Path(output_dir)
    dataset_dict = {}

    for shard_id in range(num_shards):
        shard_key = f"{split}/{shard_id}"
        shard_path = output_dir / shard_key
        dataset_dict[shard_key] = load_from_disk(str(shard_path))

    return dataset_dict

if __name__ == "__main__":
    input_dir = "."
    output_dir = "data/"
    num_shards = 12

    # --- Load dev set as a single dataset ---
    print(f"\n Creating dev set dataset...")
    # dev_path = Path(input_dir) / "dump" / "raw" / "org" / "dev"

    # def load_text_file(path):
    #     return [line.strip().split(maxsplit=1)[1] for line in open(path, encoding="utf-8")]

    # dev_dataset = Dataset.from_dict({
    #     "audio_path": load_text_file(dev_path / "wav.scp"),
    #     "text": load_text_file(dev_path / "text"),
    #     "text_ctc": load_text_file(dev_path / "text.ctc"),
    #     "text_prev": load_text_file(dev_path / "text.prev")
    # })

    # (Path(output_dir) / "dev").mkdir(parents=True, exist_ok=True)
    # dev_dataset.save_to_disk(str(Path(output_dir) / "dev"))
    # print(f"\n✅ Dev dataset saved to: {str(Path(output_dir) / 'dev')}")

    # --- Process and save training shards ---
    print(f"\n Creating train set dataset...")
    create_sharded_datasets(input_dir, output_dir, split="train", num_shards=num_shards)

    # --- Combine all datasets into a DatasetDict and save ---
    # dataset_dict_data = {"dev": load_from_disk(str(Path(output_dir) / "dev"))}
    # dataset_dict_data.update(load_sharded_datasetdict(output_dir, "train", num_shards))

    # full_dataset_dict = DatasetDict(dataset_dict_data)
    # full_dataset_dict.save_to_disk(str(Path(output_dir) / "datasetdict"))

    print(f"\n✅ DatasetDict saved to: {output_dir}")
