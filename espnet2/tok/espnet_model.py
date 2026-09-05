"""Top-level ESPnet model for speech-tokenizer training."""

from typing import Any, Dict, Mapping, Optional

import torch

from espnet2.tok.objective.abs_objective import (
    AbsSpeechTokenizerObjective,
)
from espnet2.tok.objective.asr import ASRObjective
from espnet2.tok.objective.reconstruction.hifigan import (
    HiFiGANReconstructionObjective,
)
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel


class ESPnetGANSpeechTokenizerModel(AbsGANESPnetModel):
    """Optimize a shared speech tokenizer with downstream objectives.

    The model combines ASR and speech reconstruction while keeping the objective
    boundary open to additional downstream tasks. GAN discriminator updates are
    exposed separately from the shared main loss.
    """

    def __init__(
        self,
        tokenizer: SpeechTokenizer,
        asr_objective: ASRObjective,
        reconstruction_objective: HiFiGANReconstructionObjective,
        reconstruction_weight: float = 0.1,
        extract_feats_in_collect_stats: bool = False,
        additional_objectives: Optional[
            Mapping[str, AbsSpeechTokenizerObjective]
        ] = None,
        additional_objective_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Initialize a shared tokenizer with weighted downstream objectives.

        Args:
            tokenizer: Shared SSL and differentiable-k-means tokenizer.
            asr_objective: Standard ESPnet ASR objective.
            reconstruction_objective: Adversarial waveform reconstruction.
            reconstruction_weight: Objective interpolation weight. ASR receives weight
                ``1 - alpha`` and reconstruction receives ``alpha``.
            extract_feats_in_collect_stats: Whether collect-stats must build
                this model. Speech-tokenizer inputs need shapes only.
            additional_objectives: Optional downstream objectives. Each
                consumes the same tokenizer output and training batch.
            additional_objective_weights: Loss weights keyed identically to
                ``additional_objectives``.
        """
        super().__init__()
        if not 0.0 <= reconstruction_weight <= 1.0:
            raise ValueError(
                "reconstruction_weight must be in [0, 1]: " f"{reconstruction_weight}"
            )

        additional_objectives = dict(additional_objectives or {})
        reserved_names = {"asr", "reconstruction"}
        duplicate_names = reserved_names.intersection(additional_objectives)
        if duplicate_names:
            raise ValueError(
                "Additional objective names are reserved: " f"{sorted(duplicate_names)}"
            )
        additional_weights = dict(additional_objective_weights or {})
        if set(additional_weights) != set(additional_objectives):
            raise ValueError(
                "additional_objective_weights must contain exactly the keys in "
                "additional_objectives"
            )
        if any(weight < 0.0 for weight in additional_weights.values()):
            raise ValueError("Additional objective weights must be non-negative")

        self.tokenizer = tokenizer
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.objectives = torch.nn.ModuleDict(
            {
                "asr": asr_objective,
                "reconstruction": reconstruction_objective,
                **additional_objectives,
            }
        )
        self.objective_weights = {
            "asr": 1.0 - reconstruction_weight,
            "reconstruction": reconstruction_weight,
            **additional_weights,
        }

    @property
    def asr_objective(self) -> ASRObjective:
        """Return the ASR objective."""
        return self.objectives["asr"]

    @property
    def reconstruction_objective(self) -> HiFiGANReconstructionObjective:
        """Return the adversarial reconstruction objective."""
        return self.objectives["reconstruction"]

    def _tokenize(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Run the shared tokenizer once for all main objectives."""
        return self.tokenizer(speech, speech_lengths)

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        spembs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute the weighted ASR and reconstruction main loss."""
        tokenizer_output = self._tokenize(speech, speech_lengths)
        objective_batch = dict(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            spembs=spembs,
            **kwargs,
        )

        total_loss: Optional[torch.Tensor] = None
        stats: Dict[str, Any] = {}
        for name, objective in self.objectives.items():
            objective_loss, objective_stats, _ = objective(
                tokenizer_output, **objective_batch
            )
            weighted_loss = self.objective_weights[name] * objective_loss
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )
            stats[f"{name}_weighted_loss"] = weighted_loss.detach()
            for key, value in objective_stats.items():
                stats[f"{name}_{key}"] = value

        if total_loss is None:
            raise RuntimeError("At least one main objective is required")
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
        if not self.reconstruction_objective.has_cached_generator_outputs:
            with torch.no_grad():
                tokenizer_output = self._tokenize(speech, speech_lengths)
        loss, objective_stats, weight = (
            self.reconstruction_objective.forward_discriminator(
                tokenizer_output=tokenizer_output,
                speech=speech,
                spembs=spembs,
                speech_lengths=speech_lengths,
                **kwargs,
            )
        )
        stats = {
            f"reconstruction_discriminator_{key}": value
            for key, value in objective_stats.items()
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
            if text is None or text_lengths is None:
                raise ValueError(
                    "text and text_lengths are required for generator loss"
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
        return self.asr_objective.encode(self.tokenize(speech, speech_lengths))

    def synthesize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor,
    ):
        """Tokenize speech and reconstruct waveform with speaker conditioning."""
        tokenizer_output = self.tokenize(speech, speech_lengths)
        return self.reconstruction_objective.synthesize(tokenizer_output, spembs=spembs)

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return no offline statistics; the SSL frontend runs inside the model."""
        return {}
