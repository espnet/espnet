"""ESPnet implementation of the Sidon w2v-BERT 2.0 feature predictor."""

import logging
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
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

    def _wav_to_ssl_inputs(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert raw waveforms to SSL input features on GPU.

        Replicates SeamlessM4TFeatureExtractor: kaldi fbank + per-bin CMVN
        + stride-2 frame pairing, but runs entirely on the model device.
        """
        features_list = []
        for i in range(waveforms.size(0)):
            wav = waveforms[i, : int(lengths[i].item())]
            wav = F.pad(wav, (40, 40))
            # kaldi.fbank asserts that the waveform holds at least one full
            # 25 ms analysis window, so anything shorter raises rather than
            # returning a single frame. Pad up to one window so the function
            # is total; degradations such as packet loss can leave very short
            # segments and a hard assert mid-epoch is not an acceptable
            # failure mode.
            min_window = int(round(0.025 * self.input_sr))
            if wav.numel() < min_window:
                wav = F.pad(wav, (0, min_window - wav.numel()))
            wav = wav * 32768.0  # int16 scale (SeamlessM4T convention)
            feat = kaldi.fbank(
                wav.unsqueeze(0),
                num_mel_bins=80,
                sample_frequency=float(self.input_sr),
            )
            # Per-bin CMVN matching SeamlessM4TFeatureExtractor, which uses
            # ddof=1. A very short waveform can yield a single fbank frame,
            # where the unbiased variance is NaN and would poison the whole
            # batch, so fall back to the biased estimate in that case only.
            mean = feat.mean(dim=0, keepdim=True)
            var = feat.var(dim=0, unbiased=feat.size(0) > 1, keepdim=True)
            feat = (feat - mean) / torch.sqrt(var + 1e-7)
            features_list.append(feat)

        max_len = max(f.size(0) for f in features_list)
        if max_len % 2 != 0:
            max_len += 1

        batch_size = len(features_list)
        device = waveforms.device
        input_features = torch.zeros(batch_size, max_len, 80, device=device)
        attention_mask = torch.zeros(
            batch_size, max_len, device=device, dtype=torch.long
        )
        for i, feat in enumerate(features_list):
            input_features[i, : feat.size(0)] = feat
            attention_mask[i, : feat.size(0)] = 1

        # Stride-2 frame pairing: (B, T, 80) → (B, T//2, 160)
        T = input_features.size(1)
        input_features = input_features.reshape(batch_size, T // 2, 160)
        attention_mask = attention_mask[:, 1::2]
        return {"input_features": input_features, "attention_mask": attention_mask}

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
        **kwargs,
    ):
        noisy_inputs = self.ssl_encoder._wav_to_ssl_inputs(
            noisy_speech, noisy_speech_lengths
        )
        clean_inputs = self.ssl_encoder._wav_to_ssl_inputs(
            speech_ref1, speech_ref1_lengths
        )
        predicted, _ = self.ssl_encoder(noisy_inputs)
        with torch.no_grad():
            target = self.ssl_encoder.extract_clean_features(clean_inputs)
        frames = min(predicted.size(1), target.size(1))
        # Wav2Vec2BertModel returns hidden states for padded positions too, so
        # an unmasked mse_loss averages over padding and the loss then depends
        # on batch composition. Restrict the average to frames that are valid
        # in both the degraded and the clean sequence.
        mask = (
            noisy_inputs["attention_mask"][:, :frames]
            * clean_inputs["attention_mask"][:, :frames]
        ).unsqueeze(-1)
        diff = (predicted[:, :frames].float() - target[:, :frames].float()) ** 2
        denom = mask.sum().clamp_min(1) * diff.size(-1)
        loss = (diff * mask).sum() / denom
        return force_gatherable(
            (loss, {"loss": loss.detach()}, noisy_speech.size(0)), loss.device
        )

    def collect_feats(self, **batch):
        return {}
