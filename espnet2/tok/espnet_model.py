"""Top-level ESPnet model for speech-tokenizer training."""

from typing import Any, Dict, Optional

import torch

from espnet2.tok.asr_loss import ASRLoss
from espnet2.tok.hifigan_loss import HiFiGANReconstructionLoss
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel


class ESPnetGANSpeechTokenizerModel(AbsGANESPnetModel):
    """Optimize a shared speech tokenizer with downstream losses.

    The model combines ASR and speech reconstruction. GAN discriminator updates
    are exposed separately from the shared main loss.
    """

    def __init__(
        self,
        tokenizer: SpeechTokenizer,
        asr_loss: ASRLoss,
        reconstruction_loss: HiFiGANReconstructionLoss,
        reconstruction_weight: float = 0.1,
        extract_feats_in_collect_stats: bool = False,
    ) -> None:
        """Initialize a shared tokenizer with weighted downstream losses.

        Args:
            tokenizer: Shared SSL and differentiable-k-means tokenizer.
            asr_loss: Standard ESPnet ASR loss.
            reconstruction_loss: Adversarial waveform reconstruction.
            reconstruction_weight: Loss interpolation weight. ASR receives weight
                ``1 - alpha`` and reconstruction receives ``alpha``.
            extract_feats_in_collect_stats: Whether collect-stats must build
                this model. Speech-tokenizer inputs need shapes only.
        """
        super().__init__()
        if not 0.0 <= reconstruction_weight <= 1.0:
            raise ValueError(
                "reconstruction_weight must be in [0, 1]: " f"{reconstruction_weight}"
            )

        self.tokenizer = tokenizer
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.losses = torch.nn.ModuleDict(
            {
                "asr": asr_loss,
                "reconstruction": reconstruction_loss,
            }
        )
        self.loss_weights = {
            "asr": 1.0 - reconstruction_weight,
            "reconstruction": reconstruction_weight,
        }

    @property
    def asr_loss(self) -> ASRLoss:
        """Return the ASR loss."""
        return self.losses["asr"]

    @property
    def reconstruction_loss(self) -> HiFiGANReconstructionLoss:
        """Return the adversarial reconstruction loss."""
        return self.losses["reconstruction"]

    def _tokenize(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Run the shared tokenizer once for all main losses."""
        return self.tokenizer(speech, speech_lengths)

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        spembs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute the weighted ASR and reconstruction main loss."""
        tokenizer_output = self._tokenize(speech, speech_lengths)
        loss_batch = dict(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            spembs=spembs,
            **kwargs,
        )

        total_loss: Optional[torch.Tensor] = None
        stats: Dict[str, Any] = {}
        for name, loss_module in self.losses.items():
            if self.loss_weights[name] == 0.0:
                continue
            module_loss, module_stats, _ = loss_module(tokenizer_output, **loss_batch)
            weighted_loss = self.loss_weights[name] * module_loss
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )
            stats[f"{name}_weighted_loss"] = weighted_loss.detach()
            for key, value in module_stats.items():
                stats[f"{name}_{key}"] = value

        if total_loss is None:
            raise RuntimeError("At least one main loss is required")
        stats["generator_loss"] = total_loss.detach()
        total_loss, stats, weight = force_gatherable(
            (total_loss, stats, speech.size(0)), total_loss.device
        )
        return {"loss": total_loss, "stats": stats, "weight": weight, "optim_idx": 0}

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute only discriminator loss, avoiding tokenizer work on cache hit."""
        tokenizer_output = None
        if not self.reconstruction_loss.has_cached_generator_outputs:
            with torch.no_grad():
                tokenizer_output = self._tokenize(speech, speech_lengths)
        loss, loss_stats, weight = self.reconstruction_loss.forward_discriminator(
            tokenizer_output=tokenizer_output,
            speech=speech,
            spembs=spembs,
            speech_lengths=speech_lengths,
            **kwargs,
        )
        stats = {
            f"reconstruction_discriminator_{key}": value
            for key, value in loss_stats.items()
        }
        stats["discriminator_loss"] = loss.detach()
        return {"loss": loss, "stats": stats, "weight": weight, "optim_idx": 1}

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Return main-generator or discriminator loss for its optimizer."""
        if forward_generator:
            if self.loss_weights["asr"] > 0.0 and (
                text is None or text_lengths is None
            ):
                raise ValueError(
                    "text and text_lengths are required when ASR loss is active"
                )
            return self._forward_generator(
                speech=speech,
                speech_lengths=speech_lengths,
                text=text,
                text_lengths=text_lengths,
                spembs=spembs,
                **kwargs,
            )
        return self._forward_discriminator(
            speech=speech,
            speech_lengths=speech_lengths,
            spembs=spembs,
            **kwargs,
        )

    def tokenize(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Deterministically tokenize speech for inference."""
        return self.tokenizer.encode(speech, speech_lengths)

    def encode_asr(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        """Tokenize speech and run the ASR encoder."""
        return self.asr_loss.encode(self.tokenize(speech, speech_lengths))

    def synthesize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor,
    ):
        """Tokenize speech and reconstruct waveform with speaker conditioning."""
        tokenizer_output = self.tokenize(speech, speech_lengths)
        return self.reconstruction_loss.synthesize(tokenizer_output, spembs=spembs)

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return no offline statistics; the SSL frontend runs inside the model."""
        return {}
