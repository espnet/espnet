import argparse
import json
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import kaldiio
import numpy as np
from transformers import AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser(description="process text")
    parser.add_argument("--input_path", type=Path, help="input jsonl file")
    parser.add_argument("--output_dir", type=Path, help="output directory")
    parser.add_argument("--tokenizer_tag", type=str, help="HF tokenizer tag")
    parser.add_argument("--nj", type=int, help="number of multiprocessing")
    parser.add_argument("--max_len", type=int, help="max number of tokens")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # read the whole file and make chunks
    all_lines = []
    for line in open(args.input_path, "r", encoding="utf-8", errors="ignore"):
        all_lines.append(line)

    all_chunks = []
    chunk_count, chunk_size = 0, 1000
    while chunk_count * chunk_size < len(all_lines):
        start = chunk_count * chunk_size
        end = min((chunk_count + 1) * chunk_size, len(all_lines))
        all_chunks.append(all_lines[start:end])
        chunk_count += 1

    # build tokenizer and process function
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_tag)
    process_fn = partial(process_chunk, tokenizer, args.max_len)

    # multi-process processing
    pool = Pool(args.nj)
    all_results = pool.map(process_fn, all_chunks)
    pool.close()
    pool.join()
    print("done multiprocess jobs", flush=True)

    # write kaldi scp and ark
    data_name = args.input_path.stem
    (args.output_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stats").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index_files").mkdir(parents=True, exist_ok=True)

    ark_dict = dict()
    count = 0
    for chunk in all_results:
        for ids in chunk:
            ark_dict[f"{data_name}_{count:08d}"] = ids
            count += 1

    kaldiio.save_ark(
        str(args.output_dir / "data" / f"{data_name}_tokens.ark"),
        ark_dict,
        scp=str(args.output_dir / "index_files" / "text"),
    )

    # write data.json
    data_json = {
        "task": "textlm",
        "vocabularies": [],
        "data_files": [
            str(args.output_dir / "index_files" / "text") + ",text_bpe,kaldi_ark"
        ],
        "examples": list(ark_dict.keys()),
        "num_examples": len(ark_dict),
    }
    data_json_writer = open(args.output_dir / "data.json", "wb")
    data_json_writer.write(
        json.dumps(data_json, indent=4, ensure_ascii=False, sort_keys=False).encode(
            "utf_8"
        )
    )

    # write length file
    length_writer = open(args.output_dir / "stats" / "dec_seq_shape", "w")
    for key, value in ark_dict.items():
        length_writer.write(f"textlm_{key} {len(value)}\n")
    (args.output_dir / "stats" / ".done").touch()

    print(f"Done processing {str(args.input_path)} and save to {args.output_dir}")


def process_chunk(tokenizer, max_len, iterator):
    retval = []
    for line in iterator:
        if random.random() < 0.0:
            continue
        try:
            line = json.loads(line)
        except:
            print(f"bad line: {line}", flush=True)
            continue

        if not isinstance(line, dict):
            continue

        if "text" not in line:
            continue

        all_ids = tokenizer.encode(line["text"])
        count = 0
        while count * max_len < len(all_ids):
            start = count * max_len
            end = min((count + 1) * max_len, len(all_ids))
            retval.append(np.array(all_ids[start:end], dtype=np.int32))
            count += 1

    return retval


if __name__ == "__main__":
    main()
