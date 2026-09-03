import pytest
import torch

from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)


def _attention(use_sdpa, qk_norm=False):
    torch.manual_seed(0)
    att = MultiHeadedAttention(4, 32, 0.0, qk_norm=qk_norm, use_sdpa=use_sdpa)
    return att.eval()


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_project_kv_matches_forward_qkv(use_sdpa, qk_norm):
    att = _attention(use_sdpa, qk_norm)
    q = torch.randn(3, 1, 32)
    memory = torch.randn(3, 7, 32)
    _, k_ref, v_ref = att.forward_qkv(q, memory, memory)
    k, v = att.project_kv(memory, memory)
    assert torch.equal(k, k_ref)
    assert torch.equal(v, v_ref)


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("time1", [1, 3])
@pytest.mark.parametrize("with_mask", [False, True])
def test_forward_with_kv_one_row_per_query(use_sdpa, time1, with_mask):
    """Handing the projections back must not change the result."""
    att = _attention(use_sdpa)
    q = torch.randn(5, time1, 32)
    memory = torch.randn(5, 9, 32)
    mask = (torch.rand(5, 1, 9) > 0.3) if with_mask else None
    ref = att(q, memory, memory, mask)
    out = att(q, memory, memory, mask, kv=att.project_kv(memory, memory))
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("time1", [1, 2])
@pytest.mark.parametrize("mask_rows", [None, "per_query", "per_group"])
def test_forward_with_kv_shared_over_groups(use_sdpa, time1, mask_rows):
    """One kv row per group of queries equals replicating that row.

    This is the utterance-major beam search layout: query row `b * n_rep + i`
    belongs to utterance `b`, so all `n_rep` of them attend to the same
    encoder output.
    """
    n_utt, n_rep, time2 = 2, 3, 11
    att = _attention(use_sdpa)
    q = torch.randn(n_utt * n_rep, time1, 32)
    memory = torch.randn(n_utt, time2, 32)
    replicated = memory.repeat_interleave(n_rep, dim=0)
    group_mask = torch.rand(n_utt, 1, time2) > 0.3
    group_mask[:, :, 0] = True
    if mask_rows is None:
        mask = shared_mask = None
    elif mask_rows == "per_query":
        mask = group_mask.repeat_interleave(n_rep, dim=0)
        shared_mask = mask
    else:
        mask = group_mask.repeat_interleave(n_rep, dim=0)
        shared_mask = group_mask

    ref = att(q, replicated, replicated, mask)
    out = att(q, replicated, replicated, shared_mask, kv=att.project_kv(memory, memory))
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    if not use_sdpa:
        # the attention weights keep their (batch, head, time1, time2) layout
        assert att.attn.shape == (n_utt * n_rep, 4, time1, time2)


def test_forward_with_kv_per_position_mask_falls_back():
    """A (batch, time1, time2) mask cannot be shared, so kv is expanded."""
    n_utt, n_rep, time1, time2 = 2, 2, 3, 6
    att = _attention(use_sdpa=False)
    q = torch.randn(n_utt * n_rep, time1, 32)
    memory = torch.randn(n_utt, time2, 32)
    replicated = memory.repeat_interleave(n_rep, dim=0)
    mask = torch.rand(n_utt * n_rep, time1, time2) > 0.3
    mask[:, :, 0] = True
    ref = att(q, replicated, replicated, mask)
    out = att(q, replicated, replicated, mask, kv=att.project_kv(memory, memory))
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_forward_with_kv_rejects_indivisible_groups():
    att = _attention(use_sdpa=False)
    q = torch.randn(5, 1, 32)
    memory = torch.randn(2, 6, 32)
    with pytest.raises(ValueError):
        att(q, memory, memory, None, kv=att.project_kv(memory, memory))
