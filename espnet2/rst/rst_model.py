"""Restoration feature predictor: clean SSL features from degraded speech.

This is stage 1 of Sidon (Nakata et al., arXiv:2509.17052), the reference
instance of the restoration task; the backbone is selectable.

Two self-supervised backbones are supported, selected with ``--ssl_encoder``:

* ``w2v_bert2`` -- w2v-BERT 2.0 truncated at layer 8, as in the Sidon paper
  and the official release.
* ``xeus`` -- XEUS (Chen et al., 2024), ESPnet's own E-Branchformer SSL model,
  loaded through ``SSLTask`` from the ``espnet/xeus`` release and truncated at
  block 10.

Both produce 1024-dimensional features at 50 Hz, so the vocoder stages are
unchanged. Each encoder holds a frozen teacher (features of clean speech) and
a LoRA-adapted student (features predicted from degraded speech) and exposes
the same interface, so the predictor and the vocoder never look inside:

    _wav_to_ssl_inputs(wav, lengths) -> dict     model inputs from 16 kHz audio
    encode(ssl_inputs, teacher=False) -> (features [B, T, D], frame_mask [B, T])
"""

import logging
import os
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from torch import nn

from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

logger = logging.getLogger(__name__)

# Both backbones emit one feature vector per 20 ms.
SSL_FRAME_RATE = 50


class W2VBert2Encoder(nn.Module):
    """Frozen teacher and LoRA-adapted student, both truncated at ``target_layer``."""

    def __init__(
        self,
        model_tag: str = "facebook/w2v-bert-2.0",
        target_layer: int = 8,
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
        self.target_layer = target_layer
        model_conf = dict(
            num_hidden_layers=target_layer,
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

    def _encode(
        self, model: nn.Module, ssl_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        output = model(**ssl_inputs, output_hidden_states=True)
        return output.hidden_states[self.target_layer]

    def encode(
        self, ssl_inputs: Dict[str, torch.Tensor], teacher: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Features and a 0/1 frame mask from the student or the frozen teacher."""
        if teacher:
            with torch.no_grad():
                feature = self._encode(self.teacher, ssl_inputs)
        else:
            feature = self._encode(self.student, ssl_inputs)
        return feature, ssl_inputs["attention_mask"][:, : feature.size(1)]

    def forward(
        self, ssl_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, OrderedDict]:
        feature, mask = self.encode(ssl_inputs)
        return feature, OrderedDict(pred_ssl_feat=feature, frame_mask=mask)

    @torch.no_grad()
    def extract_clean_features(
        self, ssl_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.encode(ssl_inputs, teacher=True)[0]

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


class XeusEncoder(nn.Module):
    """Frozen XEUS teacher and LoRA-adapted student, truncated at ``target_layer``.

    XEUS is loaded the ESPnet way, ``SSLTask.build_model_from_file`` on the
    ``model/config.yaml`` + ``model/xeus_checkpoint_new.pth`` pair of the
    ``espnet/xeus`` Hub release (or any directory with the same layout). The
    E-Branchformer blocks above ``target_layer`` are dropped from both copies
    since only the block-``target_layer`` output is used, and the LoRA adapter
    goes on the second linear of every feed-forward module (``w_2``, both the
    macaron and the main FFN) of the remaining blocks.

    The release is CC-BY-NC-SA-4.0; see the recipe README.
    """

    def __init__(
        self,
        model_tag: str = "espnet/xeus",
        config: str = "model/config.yaml",
        checkpoint: str = "model/xeus_checkpoint_new.pth",
        target_layer: int = 10,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Sequence[str] = ("w_2",),
        input_sr: int = 16000,
        freeze_base: bool = True,
    ):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        self.input_sr = input_sr
        self.target_layer = target_layer
        if os.path.isdir(model_tag):
            root = model_tag
        else:
            from huggingface_hub import snapshot_download

            root = snapshot_download(model_tag, allow_patterns=[config, checkpoint])
        config_path = os.path.join(root, config)
        checkpoint_path = os.path.join(root, checkpoint)

        self.teacher = self._load(config_path, checkpoint_path, target_layer)
        self.teacher.requires_grad_(False).eval()
        self.student = self._load(config_path, checkpoint_path, target_layer)
        if freeze_base:
            self.student.requires_grad_(False)
        self.student = inject_adapter_in_model(
            LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_rank,
                bias="lora_only",
                target_modules=list(lora_target_modules),
            ),
            self.student,
        )
        self._ssl_dim = self.student.encoder.output_size()
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        logger.info(
            "Sidon XEUS student: %d of %d blocks kept, trainable parameters %.2fM",
            target_layer,
            self._total_blocks,
            trainable / 1e6,
        )

    def _load(
        self, config_path: str, checkpoint_path: str, target_layer: int
    ) -> nn.Module:
        from espnet2.tasks.ssl import SSLTask

        model, _ = SSLTask.build_model_from_file(config_path, checkpoint_path, "cpu")
        self._total_blocks = len(model.encoder.encoders)
        if not 1 <= target_layer <= self._total_blocks:
            raise ValueError(
                f"target_layer must be in [1, {self._total_blocks}] for this XEUS "
                f"checkpoint, got {target_layer}"
            )
        model.encoder.encoders = model.encoder.encoders[:target_layer]
        # Training-time augmentation belongs to XEUS pretraining, not to us:
        # the student must see exactly the degraded waveform it is given.
        model.specaug = None
        return model

    @property
    def ssl_dim(self) -> int:
        return self._ssl_dim

    def _wav_to_ssl_inputs(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """XEUS consumes raw 16 kHz audio; its CNN frontend normalises it."""
        return {
            "speech": waveforms,
            "speech_lengths": lengths.to(device=waveforms.device, dtype=torch.long),
        }

    def _encode(
        self, model: nn.Module, ssl_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inference_encode: frontend -> (no SpecAugment) -> pre-encoder ->
        # encoder with every block output returned; no SSL masking unless asked.
        _, hidden_states, lengths = model.inference_encode(
            ssl_inputs["speech"],
            ssl_inputs["speech_lengths"],
            use_mask=False,
            use_final_output=False,
        )
        feature = hidden_states[self.target_layer - 1]
        mask = (~make_pad_mask(lengths, maxlen=feature.size(1))).to(
            device=feature.device, dtype=torch.long
        )
        return feature, mask

    def encode(
        self, ssl_inputs: Dict[str, torch.Tensor], teacher: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if teacher:
            with torch.no_grad():
                return self._encode(self.teacher, ssl_inputs)
        return self._encode(self.student, ssl_inputs)

    def forward(
        self, ssl_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, OrderedDict]:
        feature, mask = self.encode(ssl_inputs)
        return feature, OrderedDict(pred_ssl_feat=feature, frame_mask=mask)

    @torch.no_grad()
    def extract_clean_features(
        self, ssl_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.encode(ssl_inputs, teacher=True)[0]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self


SSL_ENCODERS = {"w2v_bert2": W2VBert2Encoder, "xeus": XeusEncoder}


def build_ssl_encoder(
    name: str,
    conf: Optional[Dict] = None,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    input_sr: int = 16000,
) -> nn.Module:
    """Instantiate the encoder named by ``--ssl_encoder`` (``--ssl_encoder_conf``)."""
    if name not in SSL_ENCODERS:
        raise ValueError(
            f"ssl_encoder must be one of {sorted(SSL_ENCODERS)}, got {name!r}"
        )
    return SSL_ENCODERS[name](
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        input_sr=input_sr,
        **dict(conf or {}),
    )


class ESPnetRestorationModel(AbsESPnetModel):
    """Predict clean SSL features from degraded speech."""

    def __init__(self, ssl_encoder: nn.Module):
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
        predicted, noisy_mask = self.ssl_encoder.encode(noisy_inputs)
        target, clean_mask = self.ssl_encoder.encode(clean_inputs, teacher=True)
        frames = min(predicted.size(1), target.size(1))
        # The encoders return hidden states for padded positions too, so an
        # unmasked mse_loss averages over padding and the loss then depends
        # on batch composition. Restrict the average to frames that are valid
        # in both the degraded and the clean sequence.
        mask = (noisy_mask[:, :frames] * clean_mask[:, :frames]).unsqueeze(-1)
        diff = (predicted[:, :frames].float() - target[:, :frames].float()) ** 2
        denom = mask.sum().clamp_min(1) * diff.size(-1)
        loss = (diff * mask).sum() / denom
        return force_gatherable(
            (loss, {"loss": loss.detach()}, noisy_speech.size(0)), loss.device
        )

    def collect_feats(self, **batch):
        return {}
