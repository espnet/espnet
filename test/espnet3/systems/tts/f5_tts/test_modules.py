"""Building blocks in ``modules.py``, exercised directly.

A full model forward only ever walks one configuration of these, so the
alternatives - the bigvgan mel front end, RMSNorm's two implementations,
attention with qk-norm or a joint context - are instantiated here on their own.
"""

import warnings

import pytest
import torch

from espnet3.systems.tts.f5_tts.modules import (
    Attention,
    AttnProcessor,
    ConvPositionEmbedding,
    MelSpec,
    RMSNorm,
    TimestepEmbedding,
    get_bigvgan_mel_spectrogram,
    get_pos_embed_indices,
    get_vocos_mel_spectrogram,
)


def _processor(**kwargs):
    """AttnProcessor warns about memory on the torch backend; not our concern."""
    kwargs.setdefault("attn_mask_enabled", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return AttnProcessor(**kwargs)


# ----------------------------------------------------------------- mel front end


def test_bigvgan_and_vocos_mels_agree_on_shape():
    """mel_spec_type picks the front end; both must stay drop-in compatible."""
    wav = torch.randn(1, 24000)

    bigvgan = get_bigvgan_mel_spectrogram(wav)
    vocos = get_vocos_mel_spectrogram(wav)

    assert bigvgan.shape[:2] == vocos.shape[:2] == (1, 100)
    assert torch.isfinite(bigvgan).all()


def test_bigvgan_mel_is_log_compressed():
    """The clamp at 1e-5 puts a hard floor under the log."""
    silence = torch.zeros(1, 24000)

    mel = get_bigvgan_mel_spectrogram(silence)

    assert torch.allclose(mel, torch.full_like(mel, torch.log(torch.tensor(1e-5))))


def test_bigvgan_mel_basis_is_cached_across_calls():
    """The basis and window are rebuilt per (shape, device) key, not per call."""
    from espnet3.systems.tts.f5_tts.modules import mel_basis_cache

    wav = torch.randn(1, 4096)
    get_bigvgan_mel_spectrogram(wav)
    size_after_first = len(mel_basis_cache)
    get_bigvgan_mel_spectrogram(wav)

    assert len(mel_basis_cache) == size_after_first


def test_melspec_dispatches_on_mel_spec_type():
    """The two front ends are not interchangeable frame for frame.

    bigvgan pads by hand and runs an uncentered STFT, vocos uses torchaudio's
    centered one, so bigvgan yields one frame fewer for the same waveform. Only
    the channel count is shared, which is why a checkpoint has to be vocoded by
    the family it was trained against.
    """
    wav = torch.randn(1, 8192)

    bigvgan = MelSpec(mel_spec_type="bigvgan")(wav)
    vocos = MelSpec(mel_spec_type="vocos")(wav)

    assert bigvgan.shape[1] == vocos.shape[1] == 100
    assert bigvgan.shape[2] == vocos.shape[2] - 1


def test_an_unknown_mel_spec_type_is_rejected():
    with pytest.raises(AssertionError):
        MelSpec(mel_spec_type="griffin_lim")


# --------------------------------------------------------------------- RMSNorm


def test_rms_norm_scales_to_unit_root_mean_square():
    norm = RMSNorm(4, eps=1e-6)
    x = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]])

    out = norm(x)

    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-4, rtol=1e-4)


def test_both_rms_norm_implementations_agree():
    """The native F.rms_norm path and the manual fallback must not diverge.

    Which one runs is decided in __init__ from the torch version, so the flag
    is flipped directly rather than by faking a version.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)

    native = RMSNorm(8, eps=1e-6)
    manual = RMSNorm(8, eps=1e-6)
    native.native_rms_norm = True
    manual.native_rms_norm = False

    torch.testing.assert_close(native(x), manual(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("native", [True, False])
def test_rms_norm_casts_the_input_to_a_half_precision_weight(native):
    """Both paths coerce x to the weight dtype so autocast does not mix them."""
    norm = RMSNorm(8, eps=1e-6)
    norm.native_rms_norm = native
    norm.weight.data = norm.weight.data.to(torch.bfloat16)

    out = norm(torch.randn(1, 2, 8))

    assert out.dtype == torch.bfloat16


# ------------------------------------------------------------------- attention


def test_attention_without_qk_norm_has_no_norm_layers():
    attn = Attention(processor=_processor(), dim=16, heads=2, dim_head=8)

    assert attn.q_norm is None and attn.k_norm is None


def test_qk_norm_adds_rms_norm_on_queries_and_keys():
    attn = Attention(
        processor=_processor(), dim=16, heads=2, dim_head=8, qk_norm="rms_norm"
    )

    assert isinstance(attn.q_norm, RMSNorm)
    assert isinstance(attn.k_norm, RMSNorm)


def test_an_unimplemented_qk_norm_is_rejected():
    with pytest.raises(ValueError, match="Unimplemented qk_norm"):
        Attention(processor=_processor(), dim=16, heads=2, dim_head=8, qk_norm="layer")


def test_a_context_dim_adds_the_joint_projections():
    """context_dim switches Attention into joint (text + audio) mode."""
    attn = Attention(
        processor=_processor(), dim=16, heads=2, dim_head=8, context_dim=12
    )

    assert attn.to_q_c.in_features == 12
    assert hasattr(attn, "to_out_c")


def test_context_pre_only_drops_the_context_output_projection():
    """The last joint layer has no further use for the context branch."""
    attn = Attention(
        processor=_processor(),
        dim=16,
        heads=2,
        dim_head=8,
        context_dim=12,
        context_pre_only=True,
    )

    assert not hasattr(attn, "to_out_c")


@pytest.mark.parametrize("qk_norm", [None, "rms_norm"])
def test_attention_forward_preserves_shape(qk_norm):
    attn = Attention(
        processor=_processor(), dim=16, heads=2, dim_head=8, qk_norm=qk_norm
    )
    x = torch.randn(2, 5, 16)

    assert attn(x).shape == x.shape


def test_masked_positions_do_not_change_the_unmasked_output():
    """Padding must not leak into the frames that matter."""
    attn = Attention(
        processor=_processor(attn_mask_enabled=True), dim=16, heads=2, dim_head=8
    ).eval()
    x = torch.randn(1, 6, 16)
    mask = torch.tensor([[True, True, True, False, False, False]])

    with torch.no_grad():
        first = attn(x, mask=mask)
        x_perturbed = x.clone()
        x_perturbed[:, 3:, :] += 10.0
        second = attn(x_perturbed, mask=mask)

    torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-5, rtol=1e-5)


def test_pe_attn_head_limits_rope_to_the_leading_heads():
    """Applying rope to a subset of heads must still return the same shape."""
    attn = Attention(processor=_processor(pe_attn_head=1), dim=16, heads=2, dim_head=8)
    x = torch.randn(1, 4, 16)
    freqs = torch.randn(1, 4, 8)  # rope acts on dim_head, not the full dim
    rope = (freqs, 1.0)

    assert attn(x, rope=rope).shape == x.shape


def test_the_torch_backend_warns_about_attention_mask_memory():
    with pytest.warns(UserWarning, match="large GPU memory"):
        AttnProcessor(attn_backend="torch", attn_mask_enabled=True)


def test_the_flash_attention_backend_requires_the_package():
    with pytest.raises(AssertionError, match="flash-attn"):
        _processor(attn_backend="flash_attn")


# --------------------------------------------------------- positional helpers


def test_pos_embed_indices_start_at_the_given_offset():
    idx = get_pos_embed_indices(torch.tensor([2]), length=3, max_pos=10)

    torch.testing.assert_close(idx, torch.tensor([[2, 3, 4]]))


def test_pos_embed_indices_are_clamped_to_max_pos():
    """Indexing past the table would be an out-of-bounds gather."""
    idx = get_pos_embed_indices(torch.tensor([8]), length=4, max_pos=10)

    assert int(idx.max()) < 10


def test_pos_embed_indices_scale_stretches_the_positions():
    plain = get_pos_embed_indices(torch.tensor([0]), length=4, max_pos=100, scale=1.0)
    stretched = get_pos_embed_indices(
        torch.tensor([0]), length=4, max_pos=100, scale=2.0
    )

    assert int(stretched.max()) > int(plain.max())


def test_conv_position_embedding_is_residual_and_shape_preserving():
    embed = ConvPositionEmbedding(dim=16)
    x = torch.randn(2, 7, 16)

    assert embed(x).shape == x.shape


def test_conv_position_embedding_zeroes_masked_frames():
    embed = ConvPositionEmbedding(dim=16)
    x = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, False, False, False]])

    out = embed(x, mask=mask)

    assert torch.all(out[:, 2:] == 0)


def test_timestep_embedding_maps_a_scalar_batch_to_features():
    embed = TimestepEmbedding(dim=16)

    out = embed(torch.tensor([0.0, 0.5, 1.0]))

    assert out.shape == (3, 16)
    # Distinct timesteps must not collapse onto the same embedding.
    assert not torch.allclose(out[0], out[2])
