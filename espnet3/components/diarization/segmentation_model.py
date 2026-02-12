"""Diarization segmentation model with powerset encoding.

Based on DiariZen architecture:
- SSL frontend (XEUS/WavLM/etc.) for feature extraction
- Projection layer
- Conformer encoder for temporal modeling
- Powerset classifier for multi-speaker segmentation
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet3.components.diarization.powerset import Powerset
from espnet3.components.diarization.ssl_frontend import SSLFrontend
from espnet3.components.diarization.pit_loss import PITLossWithPowersetEncoding


class PowersetDiarizationModel(nn.Module):
    """Powerset-based speaker diarization model.

    Architecture:
        Waveform -> SSL Frontend -> Projection -> Layer Norm ->
        Conformer Encoder -> Classifier -> Powerset Logits

    Args:
        # SSL Frontend
        ssl_model_name: Name of SSL model ("xeus", "wavlm_base", etc.)
        ssl_model_path: Path to SSL model checkpoint
        ssl_freeze: Whether to freeze SSL parameters
        ssl_num_layers: Number of SSL transformer layers
        ssl_hidden_size: SSL hidden dimension
        ssl_layer_weights: Whether to use learnable layer weights
        ssl_feature_grad_mult: Gradient multiplier for SSL features

        # Projection
        projection_size: Dimension after projection (input to conformer)

        # Conformer Encoder
        conformer_num_blocks: Number of conformer blocks
        conformer_attention_heads: Number of attention heads
        conformer_ffn_units: FFN hidden dimension
        conformer_kernel_size: Convolution kernel size
        conformer_dropout: Dropout rate

        # Powerset Classifier
        num_speakers: Maximum number of speakers
        max_speakers_per_frame: Maximum simultaneous speakers (for powerset)

        # Training
        loss_type: "nll" or "cross_entropy"
        cardinality_weight_type: "uniform", "linear", or "quadratic"
    """

    def __init__(
        self,
        # SSL Frontend
        ssl_model_name: str = "wavlm_base",
        ssl_model_path: Optional[str] = None,
        ssl_freeze: bool = False,
        ssl_num_layers: Optional[int] = None,
        ssl_hidden_size: Optional[int] = None,
        ssl_layer_weights: bool = True,
        ssl_feature_grad_mult: float = 0.1,
        # Projection
        projection_size: int = 256,
        # Conformer
        conformer_num_blocks: int = 4,
        conformer_attention_heads: int = 4,
        conformer_ffn_units: int = 1024,
        conformer_kernel_size: int = 31,
        conformer_dropout: float = 0.1,
        # Powerset
        num_speakers: int = 4,
        max_speakers_per_frame: int = 2,
        # Training
        loss_type: str = "nll",
        cardinality_weight_type: str = "uniform",
        use_pit: bool = True,  # Permutation Invariant Training
    ):
        super().__init__()

        # SSL Frontend
        self.ssl_frontend = SSLFrontend(
            model_name=ssl_model_name,
            model_path=ssl_model_path,
            freeze=ssl_freeze,
            num_layers=ssl_num_layers,
            hidden_size=ssl_hidden_size,
            layer_weights=ssl_layer_weights,
            feature_grad_mult=ssl_feature_grad_mult,
        )

        ssl_output_size = self.ssl_frontend.output_size()

        # Projection layer
        self.projection = nn.Linear(ssl_output_size, projection_size)
        self.layer_norm = nn.LayerNorm(projection_size)

        # Conformer encoder
        self.conformer = ConformerEncoder(
            input_size=projection_size,
            output_size=projection_size,
            attention_heads=conformer_attention_heads,
            linear_units=conformer_ffn_units,
            num_blocks=conformer_num_blocks,
            dropout_rate=conformer_dropout,
            positional_dropout_rate=conformer_dropout,
            attention_dropout_rate=conformer_dropout,
            input_layer=None,  # No subsampling for diarization
            normalize_before=True,
            concat_after=False,
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=conformer_kernel_size,
        )

        # Powerset encoding
        self.powerset = Powerset(
            num_classes=num_speakers,
            max_set_size=max_speakers_per_frame,
        )

        # Classifier
        self.classifier = nn.Linear(
            projection_size,
            self.powerset.num_powerset_classes
        )

        # Training parameters
        self.num_speakers = num_speakers
        self.max_speakers_per_frame = max_speakers_per_frame
        self.loss_type = loss_type
        self.cardinality_weight_type = cardinality_weight_type
        self.use_pit = use_pit

        # Get cardinality weights for loss
        self.register_buffer(
            "cardinality_weights",
            self.powerset.get_cardinality_weights(cardinality_weight_type)
        )

        # Initialize PIT loss with Hungarian algorithm
        if loss_type == "nll":
            base_loss_fn = nn.NLLLoss(reduction='none')
        else:
            base_loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.pit_loss_fn = PITLossWithPowersetEncoding(
            powerset_encoder=self.powerset,
            base_loss_fn=base_loss_fn,
            use_pit=use_pit,
        )

        logging.info(
            f"PowersetDiarizationModel initialized:\n"
            f"  SSL: {ssl_model_name}\n"
            f"  SSL output: {ssl_output_size}\n"
            f"  Projection: {projection_size}\n"
            f"  Conformer: {conformer_num_blocks} blocks\n"
            f"  Speakers: {num_speakers} (max {max_speakers_per_frame} overlapping)\n"
            f"  Powerset classes: {self.powerset.num_powerset_classes}\n"
            f"  PIT: {'enabled' if use_pit else 'disabled'} (Hungarian algorithm)"
        )

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            waveform: Input waveform
                Shape: (batch, time) or (batch, channels, time)
            waveform_lengths: Waveform lengths (batch,)

        Returns:
            logits: Powerset classification logits
                Shape: (batch, frames, num_powerset_classes)
            lengths: Output sequence lengths (batch,)
        """
        # 1. SSL feature extraction
        ssl_features, lengths = self.ssl_frontend(waveform, waveform_lengths)
        # ssl_features: (batch, time, ssl_hidden_size)

        # 2. Projection and normalization
        projected = self.projection(ssl_features)  # (B, T, projection_size)
        projected = self.layer_norm(projected)

        # 3. Conformer encoding
        # ConformerEncoder expects (batch, time, features)
        # and returns (batch, time, features), lengths

        # Conformer uses ilens internally, so pass lengths directly
        # It will create masks inside the conformer.forward() method
        encoded, _, _ = self.conformer(projected, ilens=lengths)
        # encoded: (batch, time, projection_size)

        # 4. Classification
        logits = self.classifier(encoded)
        # logits: (batch, time, num_powerset_classes)

        return logits, lengths

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute diarization loss with Permutation Invariant Training (PIT).

        Uses Hungarian algorithm (O(N³)) for efficient permutation finding instead of
        brute-force enumeration (O(N!)).

        Args:
            logits: Powerset logits (batch, frames, num_powerset_classes)
            targets: Multi-label targets (batch, frames, num_speakers)
            lengths: Sequence lengths (batch,)

        Returns:
            loss: Scalar loss (averaged over batch)
            stats: Dictionary of statistics for logging
        """
        batch_size, num_frames, num_classes = targets.shape
        assert num_classes == self.num_speakers

        # Use Hungarian-based PIT loss
        loss, best_permutations = self.pit_loss_fn(
            logits,
            targets,
            lengths=lengths,
            return_permutation=True
        )

        # Compute statistics
        with torch.no_grad():
            # Get target powerset with optimal permutation (for accuracy)
            if best_permutations is not None and self.use_pit:
                # Apply best permutation to targets
                targets_permuted = torch.zeros_like(targets)
                for b in range(batch_size):
                    perm = best_permutations[b]
                    targets_permuted[b] = targets[b, :, perm]
                target_powerset = self.powerset.to_powerset(targets_permuted)
            else:
                target_powerset = self.powerset.to_powerset(targets)

            # Predicted powerset classes
            pred_powerset = torch.argmax(logits, dim=-1)

            # Frame-level accuracy
            if lengths is not None:
                from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
                mask = ~make_pad_mask(lengths)
                accuracy = ((pred_powerset == target_powerset) & mask).sum().float() / lengths.sum()
            else:
                accuracy = (pred_powerset == target_powerset).float().mean()

        stats = {
            "loss": loss.detach(),
            "accuracy": accuracy,
        }

        return loss, stats

    def inference(
        self,
        waveform: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference (predict speaker activities).

        Args:
            waveform: Input waveform (batch, time)
            waveform_lengths: Waveform lengths (batch,)

        Returns:
            speaker_activities: Multi-label speaker activities
                Shape: (batch, frames, num_speakers)
                Values in [0, 1] (probabilities)
            lengths: Output sequence lengths (batch,)
        """
        # Forward pass
        logits, lengths = self.forward(waveform, waveform_lengths)

        # Convert powerset logits to multi-label probabilities
        speaker_activities = self.powerset.to_multilabel(logits)

        return speaker_activities, lengths


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Make mask for padding.

    Args:
        lengths: Sequence lengths (batch,)
        max_len: Maximum length (if None, use max of lengths)

    Returns:
        mask: Padding mask (batch, max_len)
            True for padded positions, False for valid positions
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max()

    seq_range = torch.arange(
        0, max_len, dtype=lengths.dtype, device=lengths.device
    )
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)

    mask = seq_range >= lengths.unsqueeze(1)

    return mask


if __name__ == "__main__":
    # Test the model
    print("Testing PowersetDiarizationModel...")

    model = PowersetDiarizationModel(
        ssl_model_name="wavlm_base",
        ssl_freeze=False,
        projection_size=256,
        conformer_num_blocks=2,
        num_speakers=3,
        max_speakers_per_frame=2,
    )

    # Test forward pass
    batch_size = 2
    duration = 3  # seconds
    sample_rate = 16000
    waveform = torch.randn(batch_size, duration * sample_rate)
    lengths = torch.tensor([duration * sample_rate, duration * sample_rate // 2])

    logits, out_lengths = model(waveform, lengths)
    print(f"Input shape: {waveform.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output lengths: {out_lengths}")

    # Test loss computation
    num_frames = logits.shape[1]
    targets = torch.randint(0, 2, (batch_size, num_frames, 3)).float()
    loss, stats = model.compute_loss(logits, targets, out_lengths)
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")

    # Test inference
    activities, _ = model.inference(waveform, lengths)
    print(f"Speaker activities shape: {activities.shape}")
    print("Test passed!")
