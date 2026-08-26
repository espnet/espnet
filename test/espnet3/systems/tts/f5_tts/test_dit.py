"""The DiT backbone and its text embedding.

A recipe forward only walks the default configuration. The optional branches -
activation checkpointing, the long skip connection, packed CFG inference, and
ZipVoice-style average upsampling - are driven directly here.
"""

import warnings

import pytest
import torch

from espnet3.systems.tts.f5_tts.backbones.dit import DiT, TextEmbedding

DIM = 16
MEL_DIM = 8
TEXT_DIM = 8
VOCAB = 20


def _dit(**kwargs):
    kwargs.setdefault("depth", 1)
    kwargs.setdefault("heads", 2)
    kwargs.setdefault("dim_head", 8)
    kwargs.setdefault("ff_mult", 1)
    with warnings.catch_warnings():
        # The torch backend warns about attention-mask memory on GPU.
        warnings.simplefilter("ignore", UserWarning)
        return DiT(
            dim=DIM,
            mel_dim=MEL_DIM,
            text_dim=TEXT_DIM,
            text_num_embeds=VOCAB,
            **kwargs,
        )


def _wake(model):
    """Give a DiT non-trivial weights.

    initialize_weights zeroes proj_out, norm_out and every block's AdaLN, so a
    freshly built DiT returns exactly zeros and any output comparison would
    pass vacuously. Filling those tensors makes the forward observable.
    """
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.02)
            block.attn_norm.linear.bias.normal_(std=0.02)
        model.norm_out.linear.weight.normal_(std=0.02)
        model.norm_out.linear.bias.normal_(std=0.02)
        model.proj_out.weight.normal_(std=0.02)
        model.proj_out.bias.normal_(std=0.02)
    return model


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return dict(
        x=torch.randn(2, 6, MEL_DIM),
        cond=torch.randn(2, 6, MEL_DIM),
        text=torch.randint(0, VOCAB, (2, 4)),
        time=torch.rand(2),
    )


# -------------------------------------------------------------- TextEmbedding


def test_average_upsampling_stretches_text_to_the_audio_length():
    """ZipVoice-style late upsampling repeats each token to fill the frames."""
    embed = TextEmbedding(
        VOCAB, TEXT_DIM, mask_padding=True, average_upsampling=True
    ).eval()
    text = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        out = embed(text, seq_len=torch.tensor([8]))

    assert out.shape == (1, 8, TEXT_DIM)
    # 4 tokens over 8 frames is an exact 2x repeat, so frames pair up.
    torch.testing.assert_close(out[0, 0], out[0, 1])
    torch.testing.assert_close(out[0, 6], out[0, 7])


def test_average_upsampling_spreads_an_uneven_remainder_to_the_tail():
    """5 frames over 2 tokens: the remainder goes to the later token."""
    embed = TextEmbedding(
        VOCAB, TEXT_DIM, mask_padding=True, average_upsampling=True
    ).eval()

    with torch.no_grad():
        out = embed(torch.tensor([[1, 2]]), seq_len=torch.tensor([5]))

    assert out.shape == (1, 5, TEXT_DIM)
    # Token 0 covers 2 frames, token 1 covers 3.
    torch.testing.assert_close(out[0, 0], out[0, 1])
    torch.testing.assert_close(out[0, 3], out[0, 4])
    assert not torch.allclose(out[0, 1], out[0, 2])


def test_average_upsampling_handles_a_ragged_batch():
    """Each sample is upsampled to its own length, shorter rows left padded."""
    embed = TextEmbedding(
        VOCAB, TEXT_DIM, mask_padding=True, average_upsampling=True
    ).eval()
    text = torch.tensor([[1, 2, 3], [4, 5, -1]])

    with torch.no_grad():
        out = embed(text, seq_len=torch.tensor([6, 4]))

    assert out.shape == (2, 6, TEXT_DIM)
    assert torch.all(out[1, 4:] == 0)  # beyond this sample's length


def test_average_upsampling_requires_mask_padding():
    """Without the mask there is no way to tell real tokens from filler."""
    with pytest.raises(AssertionError, match="text_mask_padding"):
        TextEmbedding(VOCAB, TEXT_DIM, mask_padding=False, average_upsampling=True)


def test_an_integer_seq_len_is_accepted_alongside_a_tensor():
    embed = TextEmbedding(VOCAB, TEXT_DIM, average_upsampling=True).eval()

    with torch.no_grad():
        from_int = embed(torch.tensor([[1, 2]]), seq_len=4)
        from_tensor = embed(torch.tensor([[1, 2]]), seq_len=torch.tensor([4]))

    torch.testing.assert_close(from_int, from_tensor)


def test_dropped_text_embeds_to_the_filler_token():
    """Classifier-free guidance blanks the text rather than reweighting it."""
    embed = TextEmbedding(VOCAB, TEXT_DIM).eval()
    text = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        kept = embed(text, seq_len=3, drop_text=False)
        dropped = embed(text, seq_len=3, drop_text=True)

    assert not torch.allclose(kept, dropped)
    # Every dropped position is the same filler embedding.
    torch.testing.assert_close(dropped[0, 0], dropped[0, 2])


# ------------------------------------------------------------------------- DiT


def test_dit_maps_noised_mel_back_to_mel(batch):
    out = _wake(_dit())(**batch)

    assert out.shape == batch["x"].shape
    assert torch.isfinite(out).all()


def test_a_freshly_built_dit_is_the_zero_function(batch):
    """DiT zero-inits its output layers so the model starts as identity flow."""
    out = _dit()(**batch)

    assert torch.all(out == 0)


def test_checkpoint_activations_gives_the_same_output(batch):
    """Recomputation trades memory for compute; it must not change the result."""
    torch.manual_seed(0)
    plain = _wake(_dit()).eval()
    checkpointed = _dit(checkpoint_activations=True).eval()
    checkpointed.load_state_dict(plain.state_dict())

    with torch.no_grad():
        reference = plain(**batch)
        assert not torch.all(reference == 0)  # the comparison must have teeth
        torch.testing.assert_close(reference, checkpointed(**batch))


def test_the_long_skip_connection_changes_the_output(batch):
    """It concatenates the block input back on, so it cannot be a no-op."""
    torch.manual_seed(0)
    plain = _wake(_dit()).eval()
    skipped = _dit(long_skip_connection=True).eval()
    skipped.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        skipped.long_skip_connection.weight.normal_(std=0.02)

    with torch.no_grad():
        assert not torch.allclose(plain(**batch), skipped(**batch))


def test_a_scalar_time_is_broadcast_over_the_batch(batch):
    batch["time"] = torch.tensor(0.5)

    assert _wake(_dit())(**batch).shape == (2, 6, MEL_DIM)


def test_cfg_infer_packs_the_conditional_and_unconditional_passes(batch):
    """One forward returns 2b rows: cond stacked on uncond."""
    out = _wake(_dit())(**batch, cfg_infer=True)

    assert out.shape == (4, 6, MEL_DIM)


def test_the_text_embedding_cache_is_reused_then_cleared(batch):
    """Sampling calls the backbone once per step; text is embedded once."""
    model = _dit().eval()

    with torch.no_grad():
        model(**batch, cache=True)
    assert model.text_cond is not None

    model.clear_cache()
    assert model.text_cond is None and model.text_uncond is None


def test_a_padding_mask_confines_attention_to_valid_frames(batch):
    mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])

    out = _wake(_dit(attn_mask_enabled=True))(**batch, mask=mask)

    assert out.shape == batch["x"].shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("qk_norm", [None, "rms_norm"])
def test_qk_norm_is_threaded_through_to_the_blocks(batch, qk_norm):
    assert _wake(_dit(qk_norm=qk_norm))(**batch).shape == batch["x"].shape
