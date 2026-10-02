"""Differentiable k-means quantization."""

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)


class DifferentiableKMeans(AbsSpeechTokenizerQuantizer):
    """Quantize SSL features using trainable k-means centroids.

    Training uses Gumbel-Softmax straight-through assignments whose forward
    values are hard one-hot vectors. The soft backward path permits joint
    optimization of cluster centroids and the upstream SSL frontend. Its
    results are exposed through the shared speech-tokenizer output type.
    """

    @typechecked
    def __init__(
        self,
        centroid_path: Union[str, Path],
        distance_type: str = "squared_euclidean",
        sigma_squared: float = 1.0,
        temperature_init: float = 2.0,
        temperature_floor: float = 0.1,
        temperature_decay: float = 0.999995,
    ):
        """Initialize trainable centroids from an offline k-means model.

        Args:
            centroid_path: Joblib model produced by ESPnet's k-means pipeline.
                The loaded object must expose ``cluster_centers_`` with shape
                ``(num_clusters, feature_dim)``.
            distance_type: Distance used to construct assignment logits.
                Either ``"squared_euclidean"`` or ``"euclidean"``.
            sigma_squared: Scale applied to negative distance logits.
            temperature_init: Initial Gumbel-Softmax temperature.
            temperature_floor: Minimum annealed temperature.
            temperature_decay: Exponential decay applied per training forward.
        """
        super().__init__()
        if distance_type not in ("euclidean", "squared_euclidean"):
            raise ValueError(
                "distance_type must be 'euclidean' or 'squared_euclidean', "
                f"but got {distance_type!r}"
            )
        if sigma_squared <= 0.0:
            raise ValueError(f"sigma_squared must be positive: {sigma_squared}")
        if temperature_init <= 0.0:
            raise ValueError(f"temperature_init must be positive: {temperature_init}")
        if temperature_floor <= 0.0:
            raise ValueError(f"temperature_floor must be positive: {temperature_floor}")
        if temperature_floor > temperature_init:
            raise ValueError(
                "temperature_floor must not exceed temperature_init: "
                f"{temperature_floor} > {temperature_init}"
            )
        if not 0.0 < temperature_decay <= 1.0:
            raise ValueError(
                "temperature_decay must be in (0, 1], but got " f"{temperature_decay}"
            )

        centroid_path = Path(centroid_path)
        if not centroid_path.is_file():
            raise FileNotFoundError(f"No k-means model found: {centroid_path}")

        kmeans = joblib.load(centroid_path)
        if not hasattr(kmeans, "cluster_centers_"):
            raise ValueError(f"K-means model has no cluster_centers_: {centroid_path}")
        centroids = np.asarray(kmeans.cluster_centers_)
        if centroids.ndim != 2:
            raise ValueError(
                "cluster_centers_ must have shape (num_clusters, feature_dim), "
                f"but got {centroids.shape}"
            )
        if not np.issubdtype(centroids.dtype, np.floating):
            raise TypeError(
                f"cluster_centers_ must be floating point, but got {centroids.dtype}"
            )

        self.centroids = torch.nn.Parameter(torch.from_numpy(centroids).float())
        self.distance_type = distance_type
        self.sigma_squared = float(sigma_squared)
        self.temperature_init = float(temperature_init)
        self.temperature_floor = float(temperature_floor)
        self.temperature_decay = float(temperature_decay)
        self.register_buffer(
            "temperature",
            torch.tensor(float(temperature_init), dtype=torch.float32),
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @property
    def num_clusters(self) -> int:
        """Return the number of trainable centroids."""
        return self.centroids.size(0)

    @property
    def feature_dim(self) -> int:
        """Return the centroid feature dimension."""
        return self.centroids.size(1)

    def set_temperature(self, temperature: float) -> None:
        """Set the current temperature without changing annealing parameters."""
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive: {temperature}")
        self.temperature.fill_(temperature)

    def _anneal_temperature(self) -> None:
        """Update temperature from the persisted training-forward count."""
        temperature = max(
            self.temperature_floor,
            self.temperature_init
            * self.temperature_decay ** int(self.num_updates.item()),
        )
        self.temperature.fill_(temperature)

    def _distance_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Compute scaled negative squared distances to every centroid."""
        if features.dim() != 3:
            raise ValueError(f"features must have shape (B, T, D): {features.shape}")
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Feature dimension {features.size(-1)} does not match centroid "
                f"dimension {self.feature_dim}"
            )

        centroids = self.centroids.to(dtype=features.dtype)
        if self.distance_type == "euclidean":
            distances = torch.cdist(features, centroids)
        else:
            distances = (
                features.square().sum(dim=-1, keepdim=True)
                - 2.0 * torch.matmul(features, centroids.transpose(0, 1))
                + centroids.square().sum(dim=-1)
            ).clamp_min(0.0)
        return -self.sigma_squared * distances

    @staticmethod
    def _valid_mask(
        lengths: torch.Tensor, max_length: int, device: torch.device
    ) -> torch.Tensor:
        """Return a ``(B, T)`` mask for valid frontend frames."""
        return torch.arange(max_length, device=device).unsqueeze(0) < (
            lengths.to(device).unsqueeze(1)
        )

    def _build_output(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        soft_assignment: torch.Tensor,
        assignment: torch.Tensor,
    ) -> SpeechTokenizerOutput:
        """Mask padding and construct the shared tokenizer output."""
        if feature_lengths.dim() != 1 or feature_lengths.size(0) != features.size(0):
            raise ValueError(
                "feature_lengths must have shape (B,), but got "
                f"{feature_lengths.shape} for features {features.shape}"
            )
        valid = self._valid_mask(feature_lengths, features.size(1), features.device)
        assignment = assignment * valid.unsqueeze(-1).to(assignment.dtype)
        soft_assignment = soft_assignment * valid.unsqueeze(-1).to(
            soft_assignment.dtype
        )
        token_ids = assignment.argmax(dim=-1).masked_fill(~valid, 0)
        return SpeechTokenizerOutput(
            continuous=features,
            assignment=assignment,
            token_ids=token_ids,
            lengths=feature_lengths,
            soft_assignment=soft_assignment,
        )

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Sample hard straight-through assignments for differentiable training."""
        if self.training:
            self._anneal_temperature()
        logits = self._distance_logits(features)
        soft_assignment = F.gumbel_softmax(
            logits,
            tau=float(self.temperature.item()),
            hard=False,
            dim=-1,
        )
        hard_assignment = F.one_hot(
            soft_assignment.argmax(dim=-1), num_classes=self.num_clusters
        ).to(soft_assignment.dtype)
        assignment = hard_assignment - soft_assignment.detach() + soft_assignment
        output = self._build_output(
            features,
            feature_lengths,
            soft_assignment,
            assignment,
        )
        if self.training:
            self.num_updates.add_(1)
        return output

    def encode(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Apply deterministic nearest-centroid assignment without Gumbel noise."""
        logits = self._distance_logits(features)
        soft_assignment = logits.softmax(dim=-1)
        token_ids = logits.argmax(dim=-1)
        assignment = F.one_hot(token_ids, num_classes=self.num_clusters).to(
            features.dtype
        )
        return self._build_output(
            features,
            feature_lengths,
            soft_assignment,
            assignment,
        )
