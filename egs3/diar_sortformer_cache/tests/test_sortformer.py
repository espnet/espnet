import torch
from sortformer.fastconformer_encoder import FastConformerEncoder
from sortformer.model import ESPnetSortformerModel
from sortformer.preprocessor import MelSpectrogramPreprocessor
from sortformer.sort_loss import get_ats_targets, get_pil_targets
from sortformer.sortformer_modules import SortformerModules
from sortformer.transformer_encoder import TransformerEncoder


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


def _bruteforce_pil(labels, preds):
    """Reference PIL via full permutation enumeration (small S only)."""
    import itertools

    b, t, s = labels.shape
    perms = torch.tensor(list(itertools.permutations(range(s))))
    permed = labels[:, :, perms]  # (B, T, P, S)
    preds_rep = preds.unsqueeze(2).repeat(1, 1, perms.shape[0], 1)
    score = (permed * preds_rep).sum(1).sum(2)  # (B, P)
    best = score.argmax(1)
    out = torch.stack([labels[i, :, perms[best[i]]] for i in range(b)])
    return out


def test_pil_targets_permutation_robustness():
    torch.manual_seed(0)
    b, t, s = 2, 20, 4
    labels = (torch.rand(b, t, s) > 0.6).float()
    # perfect predictions in a permuted speaker order -> PIL recovers preds
    preds = labels[:, :, [2, 0, 3, 1]].clone()
    pil = get_pil_targets(labels.clone(), preds)
    assert torch.allclose(pil, preds)


def test_pil_hungarian_matches_bruteforce():
    torch.manual_seed(1)
    b, t, s = 4, 30, 3
    labels = (torch.rand(b, t, s) > 0.55).float()
    preds = torch.rand(b, t, s)
    hung = get_pil_targets(labels.clone(), preds)
    brute = _bruteforce_pil(labels.clone(), preds)
    assert torch.equal(hung, brute)


def test_ats_targets_sorted_by_arrival():
    # speaker 2 speaks first, then 0, then 1 -> ATS order = [2, 0, 1]
    t = 12
    labels = torch.zeros(1, t, 3)
    labels[0, 6:, 0] = 1.0  # spk0 arrives at 6
    labels[0, 9:, 1] = 1.0  # spk1 arrives at 9
    labels[0, 0:, 2] = 1.0  # spk2 arrives at 0
    ats = get_ats_targets(labels)
    # channel 0 should be the earliest arriver (old spk2)
    assert torch.equal(ats[0, :, 0], labels[0, :, 2])
    assert torch.equal(ats[0, :, 1], labels[0, :, 0])
    assert torch.equal(ats[0, :, 2], labels[0, :, 1])


def test_eight_speaker_loss_runs():
    from sortformer.sort_loss import SortformerHybridLoss

    torch.manual_seed(0)
    b, t, s = 2, 50, 8
    preds = torch.rand(b, t, s, requires_grad=True)
    targets = (torch.rand(b, t, s) > 0.7).float()
    lens = torch.tensor([t, t - 5])
    loss_fn = SortformerHybridLoss(num_spks=8)
    loss, ats, pil = loss_fn(preds, targets, lens)
    assert loss.numel() == 1 and torch.isfinite(loss)
    loss.backward()
    assert preds.grad is not None and torch.isfinite(preds.grad).all()


def test_sliding_window_matches_full_attention():
    """Chunked band kernel == full rel-pos attention when window >= seq length."""
    import torch
    from sortformer.fastconformer_encoder import (
        RelPositionalEncoding,
        RelPositionMultiHeadAttention,
    )
    from sortformer.sliding_window_attention import (
        LocalAttRelPositionalEncoding,
        RelPositionLocalAttention,
    )

    torch.manual_seed(0)
    h, d, t = 4, 32, 16
    x = torch.randn(1, t, d)
    full = RelPositionMultiHeadAttention(h, d, 0.0).eval()
    w = t - 1
    loc = RelPositionLocalAttention(h, d, 0.0, [w, w]).eval()
    loc.load_state_dict(full.state_dict())  # same weights
    _, pf = RelPositionalEncoding(d, 0.0, max_len=64, xscale=None)(x)
    _, pl = LocalAttRelPositionalEncoding(
        [w, w], d_model=d, dropout_rate=0.0, max_len=64, xscale=None
    )(x)
    with torch.no_grad():
        o_full = full(x, pos_emb=pf, mask=torch.zeros(1, t, t, dtype=torch.bool))
        o_loc = loc(x, pos_emb=pl, pad_mask=torch.zeros(1, t, dtype=torch.bool))
    assert torch.allclose(o_full, o_loc, atol=1e-5)


def test_local_attention_encoder_streaming_runs():
    """FastConformer with local attention runs in streaming (cache=global)."""
    pre = MelSpectrogramPreprocessor()
    enc = FastConformerEncoder(
        feat_in=80,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling_conv_channels=16,
        att_context_size=[8, 8],
    )
    mods = SortformerModules(num_spks=4, fc_d_model=32, tf_d_model=16)
    tf = TransformerEncoder(
        num_layers=2, hidden_size=16, inner_size=32, num_attention_heads=4
    )
    model = ESPnetSortformerModel(pre, enc, mods, tf, num_spk=4)
    samples = 16000 * 20
    preds, plen = model.diarize_streaming(torch.randn(1, samples))
    assert preds.shape[2] == 4 and (preds >= 0).all() and (preds <= 1).all()


def test_transformer_sliding_window_matches_full():
    """Standard band kernel == full transformer attention when window >= length."""
    import torch
    from sortformer.sliding_window_attention import (
        TransformerLocalAttention,
    )
    from sortformer.transformer_encoder import (
        TransformerMultiHeadAttention,
    )

    torch.manual_seed(0)
    hid, heads, t = 32, 4, 16
    x = torch.randn(1, t, hid)
    full = TransformerMultiHeadAttention(hid, heads, 0.0, 0.0).eval()
    loc = TransformerLocalAttention(hid, heads, 0.0, 0.0, [t - 1, t - 1]).eval()
    loc.load_state_dict(full.state_dict())
    with torch.no_grad():
        o_full = full(x, attention_mask=None)
        o_loc = loc(x, pad_mask=torch.zeros(1, t, dtype=torch.bool), n_global=0)
    assert torch.allclose(o_full, o_loc, atol=1e-5)


def test_full_model_local_offline_and_streaming():
    """Both encoders local: offline encode + streaming run end-to-end."""
    import torch

    pre = MelSpectrogramPreprocessor()
    enc = FastConformerEncoder(
        feat_in=80,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling_conv_channels=16,
        att_context_size=[8, 8],
    )
    mods = SortformerModules(num_spks=4, fc_d_model=32, tf_d_model=16)
    tf = TransformerEncoder(
        num_layers=2,
        hidden_size=16,
        inner_size=32,
        num_attention_heads=4,
        att_context_size=[8, 8],
    )
    model = ESPnetSortformerModel(pre, enc, mods, tf, num_spk=4)
    preds, plen = model.diarize(torch.randn(1, 16000 * 8))  # offline single pass
    assert preds.shape[2] == 4 and (preds >= 0).all() and (preds <= 1).all()
    preds2, _ = model.diarize_streaming(torch.randn(1, 16000 * 20))  # streaming
    assert preds2.shape[2] == 4
