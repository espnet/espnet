#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import kaldiio
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("get text embeddings")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--hf_model_tag", type=str)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser

def dump_text_emb(
    input_file: str,
    hf_model_tag: str,
    wspecifier: str,
    max_words: int,
    batch_size: int,
):
    # (1) build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_tag)
    model = AutoModelForCausalLM.from_pretrained(hf_model_tag)
    padding_side = tokenizer.padding_side

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning(f"LM inference without CUDA will be very slow")
    model = model.to(device)

    # (2) build writer
    writer = kaldiio.WriteHelper(wspecifier)

    # (3) loop on batch inference
    example_ids, contents = [], []
    lines = open(input_file).readlines()
    for idx, line in enumerate(lines, 1):
        example_id, content = line.split(maxsplit=1)

        if len(content.split()) > max_words:
            continue

        example_ids.append(example_id)
        contents.append(content)

        if idx % batch_size == 0 or idx == len(lines):

            inputs = tokenizer(
                contents, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs, output_hidden_states=True)
                hidden_states = output.hidden_states[-1]

            for i, mask in enumerate(inputs["attention_mask"]):
                length = mask.sum().item()
                if padding_side == "right":
                    emb = hidden_states[i, :length]
                else:
                    emb = hidden_states[i, -length:]
                writer[example_ids[i]] = emb.cpu().numpy()

            example_ids, contents = [], []

            logging.info(f"processed {idx} lines")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)
    dump_text_emb(**args)