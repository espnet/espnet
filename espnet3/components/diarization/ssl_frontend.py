"""SSL frontend for diarization.

Supports various self-supervised learning models:
- XEUS
- WavLM
- HuBERT
- Wav2Vec2
- etc. (via s3prl or HuggingFace)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class SSLFrontend(nn.Module):
    """Self-supervised learning frontend for diarization.

    Extracts features from raw waveform using pre-trained SSL models.
    Supports layer-wise feature extraction and weighted combination.

    Args:
        model_name: Name of the SSL model ("xeus", "wavlm_base", "wavlm_large", etc.)
        model_path: Path to model checkpoint (if not using HuggingFace)
        freeze: Whether to freeze SSL model parameters
        num_layers: Number of transformer layers in the SSL model
        hidden_size: Hidden dimension of the SSL model
        layer_weights: Whether to use learnable layer weights
        feature_grad_mult: Gradient multiplier for SSL features (for fine-tuning)
        use_s3prl: Whether to use s3prl library (if False, use HuggingFace)
    """

    def __init__(
        self,
        model_name: str = "wavlm_base",
        model_path: Optional[str] = None,
        freeze: bool = False,
        num_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        layer_weights: bool = True,
        feature_grad_mult: float = 1.0,
        use_s3prl: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.feature_grad_mult = feature_grad_mult
        self.use_s3prl = use_s3prl

        # Load SSL model
        if model_name.lower() == "xeus" or (model_path and "xeus" in model_path.lower()):
            self._load_xeus_from_checkpoint(model_path)
            self.is_xeus = True
        elif use_s3prl:
            self._load_from_s3prl(model_name, model_path)
            self.is_xeus = False
        else:
            self._load_from_huggingface(model_name, model_path)
            self.is_xeus = False

        # Determine architecture parameters
        if num_layers is None:
            num_layers = self._infer_num_layers()
        if hidden_size is None:
            hidden_size = self._infer_hidden_size()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Layer weighting (learnable combination of all layers)
        if layer_weights:
            self.weight_sum = nn.Linear(num_layers, 1, bias=False)
        else:
            self.weight_sum = None

        # Freeze SSL model if requested
        if freeze:
            self._freeze_ssl_model()

        logging.info(
            f"SSL Frontend initialized: {model_name}, "
            f"layers={num_layers}, hidden_size={hidden_size}, "
            f"freeze={freeze}, layer_weights={layer_weights}"
        )

    def _load_from_s3prl(self, model_name: str, model_path: Optional[str]):
        """Load SSL model from s3prl library."""
        try:
            from s3prl.nn import S3PRLUpstream
        except ImportError:
            raise ImportError(
                "s3prl is not installed. Please install it with: "
                "pip install s3prl"
            )

        self.upstream = S3PRLUpstream(
            model_name,
            path_or_url=model_path,
        )
        self.upstream.eval()

    def _load_from_huggingface(self, model_name: str, model_path: Optional[str]):
        """Load SSL model from HuggingFace transformers."""
        try:
            from transformers import AutoModel, AutoFeatureExtractor
        except ImportError:
            raise ImportError(
                "transformers is not installed. Please install it with: "
                "pip install transformers"
            )

        # Map model names to HuggingFace model IDs
        model_mapping = {
            "xeus": "facebook/xeus-1b",  # Placeholder - use actual model ID
            "wavlm_base": "microsoft/wavlm-base",
            "wavlm_base_plus": "microsoft/wavlm-base-plus",
            "wavlm_large": "microsoft/wavlm-large",
            "hubert_base": "facebook/hubert-base-ls960",
            "hubert_large": "facebook/hubert-large-ll60k",
            "wav2vec2_base": "facebook/wav2vec2-base",
            "wav2vec2_large": "facebook/wav2vec2-large",
        }

        model_id = model_mapping.get(model_name, model_name)
        if model_path:
            model_id = model_path

        logging.info(f"Loading SSL model from HuggingFace: {model_id}")

        self.upstream = AutoModel.from_pretrained(
            model_id,
            output_hidden_states=True,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    def _load_xeus_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load XEUS model from ESPnet checkpoint."""
        from espnet2.tasks.ssl import SSLTask

        if checkpoint_path is None:
            raise ValueError(
                "XEUS requires a checkpoint path. Please provide model_path parameter. "
                "You can download XEUS from: https://huggingface.co/espnet/xeus/tree/main"
            )

        logging.info(f"Loading XEUS model from checkpoint: {checkpoint_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.upstream, self.train_args = SSLTask.build_model_from_file(
            None,
            checkpoint_path,
            device,
        )
        self.feature_extractor = None  # XEUS doesn't need separate feature extractor

        logging.info("XEUS model loaded successfully")

    def _infer_num_layers(self) -> int:
        """Infer number of layers from model."""
        # XEUS specific
        if self.is_xeus:
            if hasattr(self.upstream, "encoder") and hasattr(self.upstream.encoder, "encoders"):
                return len(self.upstream.encoder.encoders) + 1  # +1 for CNN
            return 25  # XEUS default: 24 transformer layers + 1 CNN

        if hasattr(self.upstream, "config"):
            return self.upstream.config.num_hidden_layers + 1  # +1 for CNN features
        elif hasattr(self.upstream, "upstream"):
            # s3prl wrapper
            if hasattr(self.upstream.upstream, "model"):
                model = self.upstream.upstream.model
                if hasattr(model, "encoder"):
                    if hasattr(model.encoder, "layers"):
                        return len(model.encoder.layers) + 1
        # Default fallback
        return 13  # Default for WavLM Base

    def _infer_hidden_size(self) -> int:
        """Infer hidden size from model."""
        # XEUS specific
        if self.is_xeus:
            if hasattr(self.upstream, "encoder") and hasattr(self.upstream.encoder, "output_size"):
                return self.upstream.encoder.output_size()
            return 1024  # XEUS default: 1024-dim

        if hasattr(self.upstream, "config"):
            return self.upstream.config.hidden_size
        elif hasattr(self.upstream, "upstream"):
            # s3prl wrapper
            if hasattr(self.upstream.upstream, "model"):
                model = self.upstream.upstream.model
                if hasattr(model.encoder, "embed_dim"):
                    return model.encoder.embed_dim
        # Default fallback
        return 768  # Default for WavLM Base

    def _freeze_ssl_model(self):
        """Freeze all parameters in the SSL model."""
        for param in self.upstream.parameters():
            param.requires_grad = False
        logging.info("SSL model parameters frozen")

    def _extract_features_huggingface(
        self,
        waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using HuggingFace model.

        Args:
            waveform: Input waveform (batch, time) or (batch, channels, time)

        Returns:
            features: Stacked layer features (batch, time, hidden_size, num_layers)
            lengths: Feature sequence lengths (batch,)
        """
        # Handle multi-channel input
        if waveform.dim() == 3:
            # Average across channels for now (can be improved)
            waveform = waveform.mean(dim=1)

        # Forward pass through SSL model
        outputs = self.upstream(
            waveform,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract all hidden states
        # hidden_states is a tuple: (CNN output, layer1, layer2, ..., layerN)
        hidden_states = outputs.hidden_states

        # Stack all layers: (batch, time, hidden_size, num_layers)
        features = torch.stack(hidden_states, dim=-1)

        # Compute output lengths (assuming downsampling factor)
        # WavLM/Wav2Vec2: 320 sample hop (16kHz -> 50 Hz)
        # This is approximate, may need adjustment per model
        batch_size = waveform.shape[0]
        wav_length = waveform.shape[1]
        feat_length = features.shape[1]
        lengths = torch.full(
            (batch_size,),
            feat_length,
            dtype=torch.long,
            device=waveform.device
        )

        return features, lengths

    def _extract_features_s3prl(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using s3prl model.

        Args:
            waveform: Input waveform (batch, time)
            lengths: Waveform lengths (batch,)

        Returns:
            features: Stacked layer features (batch, time, hidden_size, num_layers)
            lengths: Feature sequence lengths (batch,)
        """
        # s3prl returns list of layer outputs
        layer_outputs, layer_lengths = self.upstream(waveform, lengths)

        # Stack all layers
        # Each layer_output: (batch, time, hidden_size)
        features = torch.stack(layer_outputs, dim=-1)  # (B, T, H, L)

        # Use lengths from last layer (all should be same)
        lengths = layer_lengths[-1]

        return features, lengths

    def _extract_features_xeus(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using XEUS model.

        Args:
            waveform: Input waveform (batch, time)
            lengths: Waveform lengths (batch,)

        Returns:
            features: Stacked layer features (batch, time, hidden_size, num_layers)
            lengths: Feature sequence lengths (batch,)
        """
        from torch.nn.utils.rnn import pad_sequence

        # Handle multi-channel input
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)

        # XEUS encode method returns all layer outputs
        # feats is a list of tensors, one per layer
        # Each tensor: (batch, time, hidden_size)
        all_layer_outputs = self.upstream.encode(
            waveform,
            lengths,
            use_mask=False,  # Don't use masking during fine-tuning
            use_final_output=False  # Get all layer outputs, not just final
        )[0]  # First element contains the layer outputs

        # Stack all layers: (batch, time, hidden_size, num_layers)
        features = torch.stack(all_layer_outputs, dim=-1)

        # Compute output lengths (XEUS uses subsampling)
        # Typically 320x subsampling (16kHz -> 50Hz)
        feat_length = features.shape[1]
        lengths = torch.full(
            (waveform.shape[0],),
            feat_length,
            dtype=torch.long,
            device=waveform.device
        )

        return features, lengths

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract SSL features from waveform.

        Args:
            waveform: Input waveform
                Shape: (batch, time) or (batch, channels, time)
            lengths: Waveform lengths (batch,)
                Only needed for s3prl models

        Returns:
            features: Weighted combination of layer features
                Shape: (batch, time, hidden_size)
            lengths: Feature sequence lengths (batch,)
        """
        # Apply gradient multiplier
        if self.feature_grad_mult != 1.0:
            waveform = GradMultiply.apply(waveform, self.feature_grad_mult)

        # Extract features from all layers
        if self.is_xeus:
            # XEUS requires lengths
            if lengths is None:
                lengths = torch.full(
                    (waveform.shape[0],),
                    waveform.shape[-1],
                    dtype=torch.long,
                    device=waveform.device
                )
            features, lengths = self._extract_features_xeus(waveform, lengths)
        elif self.use_s3prl:
            if lengths is None:
                lengths = torch.full(
                    (waveform.shape[0],),
                    waveform.shape[-1],
                    dtype=torch.long,
                    device=waveform.device
                )
            features, lengths = self._extract_features_s3prl(waveform, lengths)
        else:
            features, lengths = self._extract_features_huggingface(waveform)

        # Apply layer weighting if enabled
        if self.weight_sum is not None:
            # features: (batch, time, hidden_size, num_layers)
            # weight_sum: linear layer (num_layers -> 1)
            features = self.weight_sum(features)  # (B, T, H, 1)
            features = features.squeeze(-1)  # (B, T, H)
        else:
            # Use last layer only
            features = features[..., -1]  # (B, T, H)

        return features, lengths

    def output_size(self) -> int:
        """Return output feature dimension."""
        return self.hidden_size


class GradMultiply(torch.autograd.Function):
    """Gradient multiplier for scaling gradients during backprop."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


if __name__ == "__main__":
    # Test SSL frontend
    print("Testing SSL Frontend...")

    # Create frontend
    frontend = SSLFrontend(
        model_name="wavlm_base",
        freeze=False,
        layer_weights=True,
    )

    # Test forward pass
    batch_size = 2
    duration = 3  # seconds
    sample_rate = 16000
    waveform = torch.randn(batch_size, duration * sample_rate)

    features, lengths = frontend(waveform)
    print(f"Input shape: {waveform.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output lengths: {lengths}")
    print(f"Output size: {frontend.output_size()}")
    print("Test passed!")
