"""ESPnet implementation of the Sidon w2v-BERT 2.0 feature predictor."""

import logging
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import nn

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

logger = logging.getLogger(__name__)


class W2VBert2Encoder(nn.Module):
    """Frozen teacher and LoRA-adapted student, both truncated at layer 8."""

    target_layer = 8

    def __init__(
        self,
        model_tag: str = "facebook/w2v-bert-2.0",
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        input_sr: int = 16000,
        freeze_base: bool = True,
    ):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import Wav2Vec2BertModel

        self.input_sr = input_sr
        model_conf = dict(
            num_hidden_layers=self.target_layer,
            layerdrop=0.0,
            attn_implementation="eager",
        )
        self.teacher = Wav2Vec2BertModel.from_pretrained(model_tag, **model_conf)
        self.teacher.requires_grad_(False).eval()

        self.student = Wav2Vec2BertModel.from_pretrained(model_tag, **model_conf)
        if freeze_base:
            self.student.requires_grad_(False)
        self.student = inject_adapter_in_model(
            LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_rank,
                bias="lora_only",
                target_modules=["output_dense"],
            ),
            self.student,
        )
        self._ssl_dim = self.student.config.hidden_size
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        logger.info(
            "Sidon w2v-BERT student trainable parameters: %.2fM", trainable / 1e6
        )

    @property
    def ssl_dim(self) -> int:
        return self._ssl_dim

    @staticmethod
    def _encode(model: nn.Module, ssl_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = model(**ssl_inputs, output_hidden_states=True)
        return output.hidden_states[W2VBert2Encoder.target_layer]

    def forward(
        self, ssl_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, OrderedDict]:
        feature = self._encode(self.student, ssl_inputs)
        return feature, OrderedDict(pred_ssl_feat=feature)

    @torch.no_grad()
    def extract_clean_features(
        self, ssl_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._encode(self.teacher, ssl_inputs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self


class SidonFeaturePredictor(AbsESPnetModel):
    """Predict clean layer-8 SSL features from degraded speech."""

    def __init__(self, ssl_encoder: W2VBert2Encoder):
        super().__init__()
        self.ssl_encoder = ssl_encoder

    def forward(
        self,
        noisy_speech: torch.Tensor,
        noisy_speech_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        noisy_speech_ssl=None,
        speech_ref1_ssl=None,
        **kwargs,
    ):
        if noisy_speech_ssl is None or speech_ref1_ssl is None:
            raise ValueError(
                "Sidon requires collated noisy_speech_ssl and speech_ref1_ssl"
            )
        device = noisy_speech.device
        noisy_inputs = {
            key: value.to(device) for key, value in noisy_speech_ssl.items()
        }
        clean_inputs = {key: value.to(device) for key, value in speech_ref1_ssl.items()}
        predicted, _ = self.ssl_encoder(noisy_inputs)
        with torch.no_grad():
            target = self.ssl_encoder.extract_clean_features(clean_inputs)
        frames = min(predicted.size(1), target.size(1))
        loss = torch.nn.functional.mse_loss(
            predicted[:, :frames].float(), target[:, :frames].float()
        )
        return force_gatherable(
            (loss, {"loss": loss.detach()}, noisy_speech.size(0)), loss.device
        )

    def collect_feats(self, **batch):
        return {}
