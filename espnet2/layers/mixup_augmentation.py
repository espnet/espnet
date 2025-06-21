import torch


class MixupAugment(torch.nn.Module):
    """Mixup augmentation module for multi-label classification."""

    def __init__(self, mixup_probability: float, mixup_alpha: float = 0.8):
        super().__init__()
        self.mixup_probability = mixup_probability
        self.mixup_alpha = mixup_alpha

    def forward(
        self, speech: torch.Tensor, onehot: torch.Tensor, speech_lengths: torch.Tensor
    ):
        """Applies mixup augmentation.

        Args:
            speech: (Batch, Length)
            onehot: (Batch, n_classes)
            speech_lengths: (Batch,)

        Returns:
            speech: (Batch, Length)
            onehot: (Batch,..., n_classes)
            speech_lengths: (Batch,): Minimum of the two lengths mixed.
        """
        batch_size = speech.size(0)
        assert onehot.size(0) == batch_size
        assert speech_lengths.size(0) == batch_size
        apply_augmentation = (
            torch.rand((batch_size), device=speech.device) < self.mixup_probability
        )
        mix_lambda = (
            torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
            .sample(sample_shape=(batch_size,))
            .to(speech.device)
            .to(dtype=speech.dtype)
        )
        perm = torch.randperm(batch_size).to(speech.device)
        identity_perm = torch.arange(batch_size, device=speech.device)
        perm[~apply_augmentation] = identity_perm[~apply_augmentation]

        speech_shape = (batch_size,) + (1,) * (speech.dim() - 1)
        mix_lambda_ = mix_lambda.view(speech_shape)
        speech = mix_lambda_ * speech + (1 - mix_lambda_) * speech[perm]

        onehot_shape = (batch_size,) + (1,) * (onehot.dim() - 1)
        mix_lambda_ = mix_lambda.view(onehot_shape)
        onehot = mix_lambda_ * onehot + (1 - mix_lambda_) * onehot[perm]

        speech_lengths = torch.minimum(speech_lengths, speech_lengths[perm])
        speech = speech[:, : speech_lengths.max()]
        if onehot.dim() > 2:
            # onehot is (Batch, length, n_classes), must ensure length match
            onehot = onehot[:, : speech_lengths.max()]
        return speech, onehot, speech_lengths
