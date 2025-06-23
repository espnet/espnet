import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torchaudio
from tqdm import tqdm

wavscp_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/mtg_jamendo/wav.scp"
segments_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/mtg_jamendo/segments"
os.makedirs(os.path.dirname(segments_path), exist_ok=True)
SEGMENT_LENGTH = 10
BATCH_SIZE = 100  # Adjust batch size as needed


def process_batch(batch):
    global lock
    lines = [line.strip().split(maxsplit=1) for line in batch]
    output_lines = []
    for utt_id, wav_path in lines:
        try:
            info = torchaudio.info(wav_path)
            duration = info.num_frames / info.sample_rate
            seg_num = 0
            start = 0.0
            while start + SEGMENT_LENGTH <= duration:
                segment_id = f"{utt_id}_segment{seg_num}"
                end = start + SEGMENT_LENGTH
                output_lines.append(f"{segment_id} {utt_id} {start:.2f} {end:.2f}\n")
                start += SEGMENT_LENGTH
                seg_num += 1
        except Exception as e:
            print(f"Failed to process {utt_id} - {wav_path}: {e}")
    if output_lines:
        with lock:
            with open(segments_path, "a") as out_f:
                out_f.writelines(output_lines)


def main():
    global lock
    with open(wavscp_path, "r") as f:
        all_lines = f.readlines()

    lock = threading.Lock()

    batches = [
        all_lines[i : i + BATCH_SIZE] for i in range(0, len(all_lines), BATCH_SIZE)
    ]

    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        list(
            tqdm(
                executor.map(process_batch, batches),
                total=len(batches),
                desc="Creating segments",
            )
        )


if __name__ == "__main__":
    main()
