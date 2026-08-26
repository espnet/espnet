"""MMDiT joint-attention blocks, carried over from upstream F5-TTS.

``MMDiTBlock`` and ``JointAttnProcessor`` are vendored but not referenced by
anything in espnet3: ``backbones/`` ships only ``dit.py``. They are pinned here
so the port stays a faithful copy and so the code cannot rot unnoticed while
nothing imports it.
"""

import warnings

import pytest
import torch

from espnet3.systems.tts.f5_tts.modules import JointAttnProcessor, MMDiTBlock

DIM = 16
HEADS = 2
DIM_HEAD = 8


def _block(**kwargs):
    kwargs.setdefault("context_dim", DIM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MMDiTBlock(dim=DIM, heads=HEADS, dim_head=DIM_HEAD, **kwargs)


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    return dict(
        x=torch.randn(2, 5, DIM),  # noised audio
        c=torch.randn(2, 3, DIM),  # text context
        t=torch.randn(2, DIM),  # time embedding
    )


def test_joint_attention_returns_both_streams(inputs):
    """MMDiT keeps text and audio as separate streams through the block."""
    block = _block()

    c_out, x_out = block(**inputs)

    assert x_out.shape == inputs["x"].shape
    assert c_out.shape == inputs["c"].shape


def test_context_pre_only_drops_the_context_stream(inputs):
    """The final block has no further use for text, so it returns None."""
    block = _block(context_pre_only=True)

    c_out, x_out = block(**inputs)

    assert c_out is None
    assert x_out.shape == inputs["x"].shape


def test_a_context_dim_differing_from_dim_cannot_work(inputs):
    """context_dim is accepted but only ever usable when it equals dim.

    The block builds AdaLayerNorm(context_dim) for the text stream and
    AdaLayerNorm(dim) for the audio stream, then feeds the SAME time embedding
    to both, so the two can only agree when the dims match. Pinned as a known
    limitation of the vendored block rather than fixed, since nothing in
    espnet3 constructs it.
    """
    block = _block(context_dim=12)
    inputs["c"] = torch.randn(2, 3, 12)

    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        block(**inputs)


def test_qk_norm_runs_through_the_joint_processor(inputs):
    block = _block(qk_norm="rms_norm")

    _, x_out = block(**inputs)

    assert torch.isfinite(x_out).all()


def test_masked_joint_attention_runs(inputs):
    block = _block(attn_mask_enabled=True)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    c_mask = torch.tensor([[True, True, True], [True, True, False]])

    c_out, x_out = block(**inputs, mask=mask, c_mask=c_mask)

    assert torch.isfinite(x_out).all()
    assert torch.isfinite(c_out).all()


def test_the_block_is_differentiable(inputs):
    """A block nothing currently trains must still be trainable."""
    block = _block()

    _, x_out = block(**inputs)
    x_out.sum().backward()

    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_the_joint_processor_rejects_flash_attention_when_absent():
    with pytest.raises(AssertionError, match="flash-attn"):
        JointAttnProcessor(attn_backend="flash_attn")


def test_rotary_embeddings_apply_to_both_streams(inputs):
    """Audio and text each carry their own rope in joint attention."""
    from espnet3.systems.tts.f5_tts.rotary import RotaryEmbedding

    rope_module = RotaryEmbedding(DIM_HEAD)
    block = _block()

    c_out, x_out = block(
        **inputs,
        rope=rope_module.forward_from_seq_len(5),
        c_rope=rope_module.forward_from_seq_len(3),
    )

    assert torch.isfinite(x_out).all()
    assert torch.isfinite(c_out).all()


def test_rope_changes_the_result(inputs):
    """Position information must actually reach the attention scores."""
    from espnet3.systems.tts.f5_tts.rotary import RotaryEmbedding

    block = _block().eval()
    rope = RotaryEmbedding(DIM_HEAD).forward_from_seq_len(5)

    with torch.no_grad():
        without = block(**inputs)[1]
        with_rope = block(**inputs, rope=rope)[1]

    assert not torch.allclose(without, with_rope)
