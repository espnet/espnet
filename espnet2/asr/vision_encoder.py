# Copyright 2022 Hyukjae Kwark
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
from filelock import FileLock
import logging
import os
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from torchvision import models, transforms
import torch.nn as nn

class ResNet(AbsEncoder):
    """ResNet vision encoder module.
    Args:
        input_size: input dim
        output_size: dimension 
    """

    def __init__(
        self,
        input_size: int = 224,
        output_size: int = 512,
        pretrained: bool = True,
        fine_tune: bool = False,
    ):
        assert check_argument_types()
        assert(pretrained or fine_tune)
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-1]) 
        RESNET_EMBEDDING_DIM = 512 # ResNET-18 output embedding is dimension 512
        
        self.fine_tune = fine_tune
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoders = model.to(self.device)
        # self.pretrained_params = copy.deepcopy(model.state_dict())
        self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.projection_layer = nn.Linear(RESNET_EMBEDDING_DIM, output_size) if output_size != RESNET_EMBEDDING_DIM else nn.Identity()
        self.output_size = output_size
        self.MAX_BATCH_SIZE = 32

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward ResNET Encoder.
        Args:
            x: input tensor (B, T, H, W, C)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        self.encoders.eval()
        batch_size = x.size(0)
        utt_length = x.size(1)
        if x.size(-1) == 3:
            x = x.permute(0,1,4,2,3)
        c = x.size(2)
        h = x.size(3)
        w = x.size(4)
        x = x.reshape(batch_size * utt_length, c, h, w)
        num_splits = batch_size * utt_length // self.MAX_BATCH_SIZE + 1
        result = []
        x = x.to(self.device)
        for i in range(num_splits):
            x_seg = x[i*self.MAX_BATCH_SIZE:min(len(x), (i+1)*self.MAX_BATCH_SIZE)]
            if self.fine_tune:
                x_seg = self.transform(x_seg)
                x_seg = self.encoders(x_seg)
                x_seg = x_seg.squeeze()
            else:
                with torch.no_grad():
                    x_seg = self.transform(x_seg)
                    x_seg = self.encoders(x_seg)
                    x_seg = x_seg.squeeze()
            x_seg = self.projection_layer(x_seg)
            assert(x_seg.size(-1) == self.output_size)
            if x_seg.dim() == 1: x_seg = x_seg.unsqueeze(0)
            result.append(x_seg)
        x = torch.cat(result)
        x = x.reshape(batch_size, utt_length, self.output_size)

        return x, ilens, None

    def reload_pretrained_parameters(self):
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1]) 
        self.encoders = model
        logging.info("Pretrained ResNet-18 model parameters reloaded!")


class VisionTransformer(AbsEncoder):
    """TODO: Vision Transformer feature extraction.
    Args:
        input_size: input dim
        output_size: dimension of attention
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,

    ):
        assert check_argument_types()
        super().__init__()
        self._ouptut_size = output_size
        self.encoders = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        raise NotImplementedError("Vision Transformer yet supported!")

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.
        Args:
            x: input tensor (B, L, D)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        with torch.no_grad():
            enc_output = self.encoders(x.to(self.device))

        return enc_output, ilens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")