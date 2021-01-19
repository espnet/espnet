# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import logging
import os
import subprocess
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class FairSeqWav2Vec2Encoder(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        freeze_w2v: whether to freeze the Wav2Vec2.0 model during training
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
    """

    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        w2v_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = True,
        freeze_w2v: bool = True,
        finetune_last_n_layers: int = 0,
    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
            except Exception as e:
                print("Error: FairSeq is not installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        self.freeze_w2v = freeze_w2v
        self._output_size = output_size
        self.finetune_last_n_layers = finetune_last_n_layers

        assert (self.freeze_w2v) or (
            self.finetune_last_n_layers > 0 and not self.freeze_w2v
        ), "freeze_w2v need to be False when finetune_last_n_layers > 0."

        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]
        if isinstance(model, Wav2VecCtc):
            model = model.w2v_encoder.w2v_model

        self.encoders = model

        self.pretrained_params = model.state_dict()

        if model.cfg.encoder_embed_dim != output_size:
            self.output_linear = torch.nn.Linear(
                model.cfg.encoder_embed_dim, output_size
            )
        else:
            self.output_linear = None

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        if self.finetune_last_n_layers > 0:
            enc_outputs = self.encoders(
                xs_pad,
                masks,
                features_only=True,
                finetune_last_n_layers=self.finetune_last_n_layers,
            )
        else:
            enc_outputs = self.encoders(xs_pad, masks)
        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)
        if self.freeze_w2v:
            xs_pad = xs_pad.detach()
            masks = masks.detach()
        olens = (~masks).sum(dim=1)

        if self.output_linear is not None:
            xs_pad = self.output_linear(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


def download_w2v(url, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    model_name = url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    if not os.path.exists(model_path):
        logging.info(f"Downloading Wav2Vec model from {url}")
        command = " ".join(["wget", "-P", dir_path, url])
        _ = subprocess.Popen(command, shell=True, stdout=None).communicate()[0]
        command = " ".join(
            [
                "wget",
                "-P",
                dir_path,
                "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt",
            ]
        )
        _ = subprocess.Popen(command, shell=True, stdout=None).communicate()[0]
        logging.info(f"Wav2Vec model downloaded {model_path}")
    else:
        logging.info(f"{model_path} exists, skipping download.")

    return model_path
