#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import re
import io
import pandas as pd
from PIL import Image
from pathlib import Path

import kaldiio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from espnet2.speechlm.tokenizer.image_tokenizer import ImageTokenizer
from espnet2.fileio.read_text import read_2columns_text

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_image")

class ImageDataset(Dataset):
    def __init__(self, data_dict, resolution, resize_choice):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.resolution = resolution
        self.resize_choice = resize_choice

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        value = self.data_dict[key]

        # (1) load as RGB matrix, int numpy array
        if isinstance(value, bytes):
            img = Image.open(io.BytesIO(value))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        else:
            raise NotImplementedError(f"Unrecognized image type: {type(img)}")
        width, height = img.size

        # (3) make square
        if self.resize_choice == "center_crop":
            if width > height:
                # Landscape image
                left = (width - height) // 2
                top = 0
                right = left + height
                bottom = height
            else:
                # Portrait image
                left = 0
                top = (height - width) // 2
                right = width
                bottom = top + width
            
            # Crop the image to make it square
            img = img.crop((left, top, right, bottom))

        elif self.resize_choice == "border":
            if width > height:
                # Landscape image
                new_size = (width, width)
                paste_position = (0, (width - height) // 2)
            else:
                # Portrait image
                new_size = (height, height)
                paste_position = ((height - width) // 2, 0)
            
            # Create a new square image with white background
            square_img = Image.new('RGB', new_size, (255, 255, 255))
            
            # Paste the original image onto the square image
            square_img.paste(img, paste_position)
            img = square_img
        else:
            raise NotImplementedError(f"Unrecognized resize op: {self.resize_choice}")
        
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        img = np.array(img)

        return key, img


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice", type=str, required=True)
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument(
        "--resize_choice", 
        type=str, 
        default='center_crop',
        choices=['center_crop', 'border'],
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--vocab_file", type=Path, required=True)
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser

def parse_parquet(data_dict):
    """ Parse the parquet files if any, to obtain the byte-form image data """
    parquet_files = dict()

    for k, v in data_dict.items():
        if re.search(r'(.+\.parquet):(\d+)$', v) is not None:
            parquet_file = v.split(':')[0]
            if parquet_file not in parquet_files:
                parquet_files[parquet_file] = list()
            parquet_files[parquet_file].append(k)
    
    for parquet_file, ids in parquet_files.items():
        df = pd.read_parquet(parquet_file)

        for imgid in ids:
            _parquet_file, rowid = data_dict[imgid].split(":")
            assert parquet_file == parquet_file
            row = df.iloc[int(rowid)]
            img_bytes = row['jpg']
            data_dict[imgid] = img_bytes
    
    return data_dict

def dump_image(
    rspecifier: str,
    wspecifier: str,
    vocab_file: str,
    model_choice: str,
    model_tag: str,
    resolution: int,
    resize_choice: str,
    batch_size: int,
    num_workers: int,
    rank: int,
):
    # (1) Device
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device_id = rank % torch.cuda.device_count()
        else:
            device_id = 0
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
        logger.warning("Image tokenization with CPU can be very slow.")
        logger.warning("Change batch_size=1 for CPU tokenization")
        batch_size = 1

    # (2) Image Tokenizer Implementation
    logger.info(f"build with model_choice: {model_choice}")
    tokenizer = ImageTokenizer(
        model_choice,
        model_tag,
        device,
    )

    # (3) load all items to be tokenized; make torch dataset
    data_dict = read_2columns_text(rspecifier)
    data_dict = parse_parquet(data_dict)
    print(f'all images to tokenize: {len(data_dict)}')
    dataset = ImageDataset(
        data_dict,
        resolution=resolution,
        resize_choice=resize_choice,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=10,
    )

    # (4) Tokenization
    image_writer = kaldiio.WriteHelper(wspecifier)
    for idx, (keys, images) in enumerate(dataloader):
        
        # Tokens of size: [B, W, H] or [B, W, H, n_q]
        tokens = tokenizer(images).flatten(start_dim=1)
        tokens_numpy = tokens.cpu().numpy()

        for key, token in zip(keys, tokens_numpy):
            image_writer[key] = token
        
        if idx > 0 and idx % 10 == 0:
            print(f'done {idx} batches', flush=True)
        
    # (4) dump vocabulary file and image_code_per_frame file
    if rank == 1:
        vocab_writer = open(vocab_file, "w")
        for codebook_idx in range(tokenizer.n_codebook):
            for code_idx in range(tokenizer.size_codebook):
                vocab_writer.write(f"<image_layer{codebook_idx}_code{code_idx}>\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)
    dump_image(**args)
