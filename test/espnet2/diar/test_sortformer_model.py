import torch

from espnet2.diar.espnet_sortformer_model import ESPnetSortformerModel
from espnet2.diar.sortformer.fastconformer_encoder import FastConformerEncoder
from espnet2.diar.sortformer.preprocessor import MelSpectrogramPreprocessor
from espnet2.diar.sortformer.sort_loss import get_ats_targets, get_pil_targets
from espnet2.diar.sortformer.sortformer_modules import SortformerModules
from espnet2.diar.sortformer.transformer_encoder import TransformerEncoder


def _tiny_model(num_spk=4):
    pre = MelSpectrogramPreprocessor()
    enc = FastConformerEncoder(
        feat_in=80,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling_conv_channels=16,
        conv_kernel_size=9,
    )
    mods = SortformerModules(num_spks=num_spk, fc_d_model=32, tf_d_model=16)
    tf = TransformerEncoder(
        num_layers=2, hidden_size=16, inner_size=32, num_attention_heads=4
    )
    return ESPnetSortformerModel(pre, enc, mods, tf, num_spk=num_spk)


def test_forward_train_returns_loss_stats_weight():
    torch.manual_seed(0)
    model = _tiny_model()
    b, samples = 2, 16000  # 1 second
    speech = torch.randn(b, samples)
    lengths = torch.tensor([samples, samples - 2000])
    # frame count ~ ceil(ceil(samples/160)/8)
    n_frames = -(-(-(-samples // 160)) // 8)
    spk_labels = (torch.rand(b, n_frames, 4) > 0.7).float()
    spk_lens = torch.tensor([n_frames, n_frames - 5])
    loss, stats, weight = model(speech, lengths, spk_labels, spk_lens)
    assert loss.numel() == 1 and torch.isfinite(loss).all()
    for k in ("loss", "ats_loss", "pil_loss", "f1_acc"):
        assert k in stats
    assert int(weight) == b


def test_encoder_subsamples_8x():
    model = _tiny_model()
    speech = torch.randn(1, 16000)
    preds, plen = model.diarize(speech)
    # ~100 mel frames / 8 ~= 12-13 output frames
    assert preds.shape[0] == 1 and preds.shape[2] == 4
    assert 8 <= preds.shape[1] <= 16
    assert (preds >= 0).all() and (preds <= 1).all()


def test_ats_and_pil_targets_permutation_robustness():
    torch.manual_seed(0)
    b, t, s = 2, 20, 4
    labels = (torch.rand(b, t, s) > 0.6).float()
    # perfect predictions in a permuted speaker order -> targets recover labels
    perm = [2, 0, 3, 1]
    preds = labels[:, :, perm].clone()
    import itertools

    perms = torch.tensor(list(itertools.permutations(range(s))))
    pil = get_pil_targets(labels.clone(), preds, perms)
    # PIL target should match preds (the permutation that best matches preds)
    assert torch.allclose(pil, preds)
    ats = get_ats_targets(labels.clone(), preds, perms)
    assert ats.shape == labels.shape
