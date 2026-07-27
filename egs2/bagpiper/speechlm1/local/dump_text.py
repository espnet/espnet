#!/usr/bin/env python3
"""Dump text from JSONL files with multiprocessing support."""

import argparse
import io
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import zstandard as zstd

from espnet2.speechlm.utils.parquet_dump import ArkiveWriter

def parse_kaldi(line: str, idx: int) -> tuple:
    uttid, content = line.strip().split(maxsplit=1)
    return uttid, content

def parse_qwen_caption(line: str, idx: int) -> tuple:
    """Parse a single line for qwen_caption mode.

    Args:
        line: A JSON string containing utt_id and caption fields.

    Returns:
        Tuple of (utt_id, caption) or (None, None) if parsing fails.
    """
    try:
        data = json.loads(line.strip())
        # Check finish_reason is "stop"
        if data.get("finish_reason") != "stop":
            return None, None
        utt_id = data.get("utt_id")
        caption = data.get("caption")
        if utt_id is not None and caption is not None:
            return utt_id, caption
    except json.JSONDecodeError:
        pass
    return None, None


def parse_dolma3(line: str, idx: int) -> tuple:
    """Parse a single line for dolma3 mode.

    Args:
        line: A JSON string containing id and text fields.

    Returns:
        Tuple of (utt_id, caption) or (None, None) if parsing fails.
    """
    try:
        data = json.loads(line.strip())
        utt_id = data.get("id", None)
        if utt_id is None:
            utt_id = data.get("warc_record_id", None)
        caption = data.get("text")
        if utt_id is not None and caption is not None:
            return utt_id, caption
    except json.JSONDecodeError:
        pass
    return None, None

def parse_llama_nemotron(line: str, example_id: int) -> tuple:
    data = json.loads(line.strip())
    messages = list()
    for msg in data['input']:
        role = msg['role']
        content = msg['content']

        assert role in ['system', 'user', 'assistant'], f"Unexpected role: {role}"
        messages.append((role, "text", content))
    
    content = data['output']
    messages.append(("assistant", "text", content))

    text = json.dumps(messages)
    return example_id, text

def open_file(file_path: str):
    """Open a file, handling zstd compression if needed.

    Args:
        file_path: Path to the file.

    Returns:
        A file-like object for reading lines.
    """
    if file_path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        fh = open(file_path, "rb")
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding="utf-8")
    else:
        return open(file_path, "r", encoding="utf-8")


def process_single_file(args: tuple) -> dict:
    """Process a single JSONL file (optionally zstd-compressed).

    Args:
        args: Tuple of (file_path, mode).

    Returns:
        Dictionary mapping utt_id to caption.
    """
    file_path, mode = args
    data_dict = {}

    if mode == "qwen_caption":
        parse_func = parse_qwen_caption
    elif mode == "dolma3":
        parse_func = parse_dolma3
    elif mode == "llama_nemotron":
        parse_func = parse_llama_nemotron
    elif mode == "kaldi":
        parse_func = parse_kaldi
    else:
        raise ValueError(f"Unknown mode: {mode}")

    file_stem = Path(file_path).stem
    with open_file(file_path) as f:
        for idx, line in enumerate(f):
            if line.strip():
                utt_id, caption = parse_func(line, f"{file_stem}_{idx}")
                if utt_id is not None and caption is not None:
                    data_dict[utt_id] = caption

    print(f"File processing {file_path} is done. Get samples: {len(data_dict)}", flush=True)
    return data_dict


def process_files_batch(file_paths: list, mode: str, num_workers: int) -> dict:
    """Process a batch of files using multiprocessing.

    Args:
        file_paths: List of file paths to process.
        mode: Parsing mode (e.g., "qwen_caption").
        num_workers: Number of worker processes.

    Returns:
        Combined dictionary mapping utt_id to caption.
    """
    args_list = [(fp, mode) for fp in file_paths]

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, args_list)

    # Combine all results into a single dictionary
    combined_dict = {}
    for idx, result in enumerate(results):
        combined_dict.update(result)
        print(f'Rank {idx}: finished: {len(combined_dict)}')

    print('aggregate all results', flush=True)
    return combined_dict


def main():
    parser = argparse.ArgumentParser(
        description="Dump text from JSONL files with multiprocessing support."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dumped text files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="qwen_caption",
        help="Parsing mode (default: qwen_caption).",
    )
    parser.add_argument(
        "--file_regex",
        type=str,
        default=r".*\.jsonl$",
        help="Regex pattern to match files (default: .*\\.jsonl$).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Number of worker processes (default: 128).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Chunk size for ArkiveWriter (default: 50000).",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Recursively find all files matching the regex pattern
    pattern = re.compile(args.file_regex)
    input_path = Path(args.input_dir)
    matched_files = sorted(
        str(f) for f in input_path.rglob("*") if f.is_file() and pattern.search(f.name)
    )

    if not matched_files:
        print(f"No files matching '{args.file_regex}' found in {args.input_dir}")
        return

    print(f"Found {len(matched_files)} files matching '{args.file_regex}' in {args.input_dir}")

    # Create ArkiveWriter
    writer = ArkiveWriter(
        output_dir=args.output_dir,
        data_name=input_path.stem,
        data_type="text",
        target_format="string",
        chunk_size=args.chunk_size,
        max_workers=args.num_workers,
    )

    # Process files in batches of num_workers
    batch_size = args.num_workers * 100
    total_entries = 0

    for i in range(0, len(matched_files), batch_size):
        batch_files = matched_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: files {i + 1} to {i + len(batch_files)}")

        batch_data = process_files_batch(batch_files, args.mode, args.num_workers)
        total_entries += len(batch_data)

        print(f"  Collected {len(batch_data)} entries from this batch")

        # Write batch data to arkive
        writer.write(batch_data)

    # Finalize the writer (flush remaining buffer and merge metadata)
    writer.finalize()

    print(f"Total entries collected: {total_entries}")
    print(f"Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
