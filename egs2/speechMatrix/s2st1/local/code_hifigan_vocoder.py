"""
From https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/text_to_speech/vocoder.py
With CPU model loading fixed
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Dict

import torch
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.text_to_speech.codehifigan import CodeGenerator as CodeHiFiGANModel
from fairseq.models.text_to_speech.hifigan import Generator as HiFiGANModel
from torch import nn

logger = logging.getLogger(__name__)


class HiFiGANVocoder(nn.Module):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = HiFiGANModel(model_cfg)
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict["generator"])
        if fp16:
            self.model.half()
        logger.info(f"loaded HiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B x) T x D -> (B x) 1 x T
        model = self.model.eval()
        if len(x.shape) == 2:
            return model(x.unsqueeze(0).transpose(1, 2)).detach().squeeze(0)
        else:
            return model(x.transpose(-1, -2)).detach()

    @classmethod
    def from_data_cfg(cls, args, data_cfg: S2TDataConfig):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg.get("type", "griffin_lim") == "hifigan"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)


@register_model("CodeHiFiGANVocoder")
class CodeHiFiGANVocoder(BaseFairseqModel):
    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: Dict[str, str],
        fp16: bool = False,
        cpu: bool = False,
    ) -> None:
        super().__init__()
        self.model = CodeHiFiGANModel(model_cfg)
        if cpu:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        logger.info(f"loaded CodeHiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        return self.model(**x).detach().squeeze()

    @classmethod
    def from_data_cfg(cls, args, data_cfg):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg is not None, "vocoder not specified in the data config"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)
