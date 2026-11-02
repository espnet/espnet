import pytest
import torch

from espnet2.layers.label_aggregation import LabelAggregate
from espnet2.layers.stft import Stft

# (win_length, hop_length): default diar frontend, 25 ms / 10 ms at 16 kHz,
# and the enh_diar recipes (ConvTasNet stride x subsampling).
SETTINGS = [(512, 128), (400, 160), (256, 64)]


def _labels(lengths, n_spk=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Piecewise-constant activity, like real RTTM-derived labels.
    x = torch.zeros(len(lengths), max(lengths), n_spk)
    for b, n in enumerate(lengths):
        for s in range(n_spk):
            start, end = sorted(torch.randint(0, n, (2,), generator=g).tolist())
            x[b, start:end, s] = 1.0
    return x


@pytest.mark.parametrize(
    ("input_label", "expected_output"),
    [
        (torch.ones(10, 20000, 2), torch.ones(10, 157, 2)),
        (torch.zeros(10, 20000, 2), torch.zeros(10, 157, 2)),
    ],
)
def test_LabelAggregate(input_label, expected_output):
    label_aggregate = LabelAggregate(win_length=512, hop_length=128, center=True)
    aggregated_label, _ = label_aggregate.forward(input=input_label)
    assert torch.equal(aggregated_label, expected_output)


def test_LabelAggregate_repr():
    label_aggregate = LabelAggregate(win_length=512, hop_length=128, center=True)
    print(label_aggregate)


def test_LabelAggregate_no_center():
    win_length, hop_length = 512, 128
    input_label = torch.zeros(2, 2000, 2)
    input_label[0, :1000, 0] = 1.0
    input_label[1, 300:, 1] = 1.0
    ilens = torch.tensor([2000, 1500])

    label_aggregate = LabelAggregate(
        win_length=win_length, hop_length=hop_length, center=False
    )
    aggregated_label, olens = label_aggregate.forward(input=input_label, ilens=ilens)

    # Same framing as torch.stft(..., center=False)
    frames = input_label.unfold(1, win_length, hop_length)
    expected = (frames.sum(dim=-1) > win_length // 2).float()
    expected_olens = (ilens - win_length) // hop_length + 1
    for b, olen in enumerate(expected_olens):
        expected[b, olen:] = 0.0

    assert aggregated_label.shape == (2, 12, 2)
    assert torch.equal(olens, expected_olens)
    assert torch.equal(aggregated_label, expected)


@pytest.mark.parametrize("win_length, hop_length", SETTINGS)
@pytest.mark.parametrize("center", [True, False])
def test_frames_match_stft(win_length, hop_length, center):
    # Lengths: off-grid, exactly one window, exactly on the hop grid.
    lengths = [16000, 12345, win_length, win_length + 3 * hop_length]
    x = _labels(lengths)
    ilens = torch.tensor(lengths)

    labels, olens = LabelAggregate(win_length, hop_length, center)(x.clone(), ilens)
    stft = Stft(n_fft=win_length, hop_length=hop_length, center=center)
    feats, flens = stft(torch.randn(len(lengths), max(lengths)), ilens)

    assert labels.shape[1] == feats.shape[1]
    assert torch.equal(olens, flens)
    assert ((labels == 0) | (labels == 1)).all()


@pytest.mark.parametrize("win_length, hop_length", SETTINGS)
def test_no_center_matches_unfold(win_length, hop_length):
    lengths = [16000, 12345]
    x = _labels(lengths, seed=1)
    labels, olens = LabelAggregate(win_length, hop_length, center=False)(
        x.clone(), torch.tensor(lengths)
    )
    expected = (x.unfold(1, win_length, hop_length).sum(-1) > win_length // 2).float()
    for b, n in enumerate(olens):
        expected[b, n:] = 0.0
    assert torch.equal(labels, expected)


@pytest.mark.parametrize("center", [True, False])
def test_ilens_none(center):
    x = _labels([8000, 8000])
    with_lens, _ = LabelAggregate(512, 128, center)(
        x.clone(), torch.tensor([8000, 8000])
    )
    without, olens = LabelAggregate(512, 128, center)(x.clone())
    assert olens is None
    assert torch.equal(with_lens, without)


@pytest.mark.parametrize("center", [True, False])
def test_non_contiguous_input(center):
    x = _labels([8000, 6000], n_spk=3)
    non_contig = x.transpose(1, 2).contiguous().transpose(1, 2)
    assert not non_contig.is_contiguous()
    ref, _ = LabelAggregate(512, 128, center)(x.clone())
    out, _ = LabelAggregate(512, 128, center)(non_contig)
    assert torch.equal(out, ref)
