import torch


class MixupAugment(torch.nn.Module):
    """Mixup augmentation module for multi-label classification."""

    def __init__(self, mixup_probability: float):
        super().__init__()
        self.mixup_probability = mixup_probability

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
            onehot: (Batch, n_classes)
            speech_lengths: (Batch,): Minimum of the two lengths mixed.
        """
        batch_size = speech.size(0)
        assert onehot.size(0) == batch_size
        assert speech_lengths.size(0) == batch_size
        apply_augmentation = (
            torch.rand((batch_size), device=speech.device) < self.mixup_probability
        )
        mix_lambda = (
            torch.distributions.Beta(0.8, 0.8)
            .sample(sample_shape=(batch_size, 1))
            .to(speech.device)
        )
        perm = torch.randperm(batch_size).to(speech.device)
        identity_perm = torch.arange(batch_size, device=speech.device)
        perm[~apply_augmentation] = identity_perm[~apply_augmentation]
        speech = mix_lambda * speech + (1 - mix_lambda) * speech[perm]
        onehot = mix_lambda * onehot + (1 - mix_lambda) * onehot[perm]
        speech_lengths = torch.minimum(speech_lengths, speech_lengths[perm])
        return speech, onehot, speech_lengths
