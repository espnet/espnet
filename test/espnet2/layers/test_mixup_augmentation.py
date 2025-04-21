import torch

from espnet2.layers.mixup_augmentation import MixupAugment


def test_mixup_augment_shape():
    """Basic test to check shape preservation."""
    speech = torch.randn(7, 11)  # (Batch, Length)
    label = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1]]
    )  # (Batch, n_classes)
    speech_lengths = torch.tensor([11] * 7, dtype=torch.long)  # (Batch,)
    mixup = MixupAugment(mixup_probability=1.0)
    speech_aug, label_aug, speech_lengths_aug = mixup(speech, label, speech_lengths)
    assert speech_aug.shape == speech.shape
    assert label_aug.shape == label.shape
    assert speech_lengths_aug.shape == speech_lengths.shape


def test_mixup_noop():
    """Test when mixup_prob = 0.0 to ensure no change in data."""
    speech = torch.randn(6, 10)  # (Batch, Length)
    label = torch.eye(6)  # One-hot labels (Batch, n_classes)
    speech_lengths = torch.tensor([10] * 6, dtype=torch.long)  # (Batch,)

    mixup = MixupAugment(mixup_probability=0.0)
    (speech_aug, label_aug, speech_lengths_aug) = mixup(speech, label, speech_lengths)

    assert torch.allclose(label_aug, label, atol=1e-4)
    assert torch.allclose(speech_aug, speech, atol=1e-4)
    assert torch.all(speech_lengths == speech_lengths_aug)


def test_mixup_consistency():
    speech = torch.randn(2, 25)  # (B, L)
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (B, n_classes)
    speech_lengths = torch.tensor([25, 25], dtype=torch.long)  # (B,)
    mixup = MixupAugment(mixup_probability=1.0)
    # NOTE(shikhar) False negatives: When permutation inside
    # mixuaugment was [0,1], the test will pass but when it
    # fails, there's a bug.
    speech_aug, label_aug, speech_lengths_aug = mixup(speech, label, speech_lengths)
    lam = [label_aug[0, 0].item(), label_aug[1, 1].item()]
    expected_speech = torch.zeros_like(speech)
    expected_speech[0] = lam[0] * speech[0] + (1 - lam[0]) * speech[1]
    expected_speech[1] = lam[1] * speech[1] + (1 - lam[1]) * speech[0]
    assert torch.allclose(speech_aug, expected_speech, atol=1e-4)
    assert torch.all(speech_lengths_aug == 25)
