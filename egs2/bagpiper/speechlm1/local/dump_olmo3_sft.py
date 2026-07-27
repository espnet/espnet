#!/usr/bin/env python3
"""Dump OLMO-3 SFT data from HuggingFace."""

import argparse
import os
import json

from datasets import load_dataset
from espnet2.speechlm.utils.parquet_dump import ArkiveWriter

def process_fn(example):
    id = example.pop("id")
    messages = []

    for msg in example['messages']:
        content = msg['content']
        role = msg['role']
        modality = "text"
        
        if role not in ["assistant", "user", "system"]:
            return None
        
        if not isinstance(content, str):
            return None 

        messages.append((role, modality, content))
    
    return id, json.dumps(messages)

def main():
    parser = argparse.ArgumentParser(
        description="Dump OLMO-3 SFT data from HuggingFace."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/Dolci-Think-SFT-32B",
        help="HuggingFace dataset tag (default: allenai/Dolci-Think-SFT-32B).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dumped data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the HuggingFace dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split, num_proc=8)
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    writer = ArkiveWriter(
        output_dir=args.output_dir,
        data_name=args.dataset.replace("/", "_"),
        data_type="text",
        target_format="string",
        chunk_size=50000,
        max_workers=32,
    )

    # Iterator loop over the dataset
    data_dict = dict()
    for idx, example in enumerate(dataset, 1):
        
        result = process_fn(example)
        if result is None:
            continue

        id, messages = result
        data_dict[id] = messages

        if idx % 1e6 == 0 or idx == len(dataset):
            writer.write(data_dict)
            data_dict = dict()
    
    writer.finalize()

if __name__ == "__main__":
    main()
