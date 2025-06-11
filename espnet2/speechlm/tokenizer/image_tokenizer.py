#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys

import numpy as np
import torch

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer


class ImageTokenizer(AbsTokenizer):
    """Image Tokenizer implementation

    Use cases:
        - use encode and decode for discrete (de)tokenization
    """

    def __init__(
        self,
        model_choice: str,
        model_tag: int,
        device: str,
    ):
        """Image Tokenizer initialization

        Each of the codec implementation should assign all following features:
            self.n_codebook (int): the number of codec codebooks.
            self.size_codebook (int): the dimension of codebooks.
        """

        super().__init__()
        self.model_choice = model_choice
        self.model_tag = model_tag
        self.device = device

        if self.model_choice == "cosmos":
            model_dir = os.path.join("Cosmos-Tokenizer", "ckpt")
            config_file1 = os.path.join(model_dir, "model_config.yaml")
            config_file2 = os.path.join(model_dir, "config.json")

            if not (os.path.exists(config_file1) or os.path.exists(config_file2)):
                raise ValueError(
                    f"Haven't found the target checkpoint file. \n"
                    f"To use cosmos tokenizer, please do as follow \n"
                    f"git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git \n"
                    f"huggingface-cli download --repo-type model --local-dir "
                    f"./Cosmos-Tokenizer/ckpt nvidia/${model_tag} "
                )

            # TODO(Jinchuan): put this operation into shell script
            sys.path.append("./Cosmos-Tokenizer")
            from cosmos_tokenizer.image_lib import ImageTokenizer
            from cosmos_tokenizer.utils import numpy2tensor as cosmos_numpy2tensor
            from cosmos_tokenizer.utils import tensor2numpy as cosmos_tensor2numpy

            self.tokenizer = ImageTokenizer(
                checkpoint_enc=os.path.join(model_dir, "encoder.jit"),
                checkpoint_dec=os.path.join(model_dir, "decoder.jit"),
                device=device,
                dtype="bfloat16",
            )
            self.cosmos_numpy2tensor = cosmos_numpy2tensor
            self.cosmos_tensor2numpy = cosmos_tensor2numpy

            # NOTE(Jinchuan) Cannot parse from the model; use hard code
            self.n_codebook = 1
            self.size_codebook = 64000  # FSQ: (8, 8, 8, 5, 5, 5)

        elif self.model_choice == "vila-u":
            config_file = os.path.join("vila-u", "ckpt", "config.json")
            ckpt_dir = os.path.join("vila-u", "ckpt", "vision_tower")

            if not os.path.exists(config_file) or not os.path.exists(ckpt_dir):
                raise ValueError(
                    f"Haven't found the target checkpoint file. \n"
                    f"To use vila-u tokenizer, please do as follow \n"
                    f"git clone https://github.com/mit-han-lab/vila-u \n"
                    f"huggingface-cli download --repo-type model --local-dir "
                    f"./vila-u/ckpt mit-han-lab/vila-u-7b-256 "
                )

            # TODO(Jinchuan): put this operation into shell script
            sys.path.append("./vila-u")
            from transformers import PretrainedConfig
            from vila_u.model.multimodal_encoder.builder import build_vision_tower

            config = PretrainedConfig.from_json_file(config_file)
            model = build_vision_tower(ckpt_dir, config)
            self.tokenizer = model.vision_tower.rqvaesiglip.to(device).to(
                torch.bfloat16
            )
            self.processor = model.image_processor

            self.n_codebook = 4
            self.size_codebook = 16384

        else:
            raise ValueError(f"Image tokenizer {model_choice} is not supported")

    @torch.no_grad()
    def forward(self, images):
        """Convert image to token"""
        if self.model_choice == "cosmos":
            assert images.dtype == torch.uint8
            images = self.cosmos_numpy2tensor(images.cpu().numpy())
            tokens = self.tokenizer.encode(images)[0]
            return tokens

        if self.model_choice == "vila-u":
            images = images.cpu().numpy()
            images = torch.stack(
                [
                    self.processor(img, return_tensors="pt")["pixel_values"][0]
                    for img in images
                ],
                dim=0,
            )
            images = images.to(torch.bfloat16).to(self.device)
            tokens = self.tokenizer.encode_image(images)[0].int()

            shift = torch.arange(self.n_codebook).to(self.device)
            tokens += shift.view(1, 1, 1, -1) * self.size_codebook

            return tokens

        else:
            raise NotImplementedError

    @torch.no_grad()
    def detokenize(self, codes):
        """
        Reconstruct images from tokenizer codes
        Input (torch.Tensor): [B, W * H * C], torch.long
        Output (torch.Tensor): [B, W, H, C], torch.uint8
        """
        if self.model_choice == "cosmos":
            assert codes.dim() == 2
            size = int((codes.size(1) / self.n_codebook) ** 0.5)
            codes = codes.view(-1, size, size)

            rec_images = self.tokenizer.decode(codes)
            rec_images = torch.from_numpy(self.cosmos_tensor2numpy(rec_images))

            return rec_images

        elif self.model_choice == "vila-u":
            # resize to [B, H, W, C]. Will raise an error if size is not compatible
            assert codes.dim() == 2
            size = int((codes.size(1) / self.n_codebook) ** 0.5)
            codes = codes.view(-1, size, size, self.n_codebook)

            shift = torch.arange(self.n_codebook).to(self.device)
            codes -= shift.view(1, 1, 1, -1) * self.size_codebook

            # convert back to range [0, 255]
            rec_images = self.tokenizer.decode(
                self.tokenizer.quantizer.embed_code(codes)
            ).float()
            rec_images = rec_images.add_(1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
            rec_images = rec_images.permute(0, 2, 3, 1)

            return rec_images
        else:
            raise NotImplementedError
