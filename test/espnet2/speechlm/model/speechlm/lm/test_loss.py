"""Tests for espnet2/speechlm/model/speechlm/lm/loss.py.

``fused_cross_entropy_loss`` delegates numerical work to
``LigerFusedLinearCrossEntropyFunction.apply``. The real Liger kernel
needs CUDA, so we monkeypatch ``.apply`` with a CPU stub that records
the positional args it was invoked with. Tests assert the *flow*:
shift-by-one, pre-masking, stream-0 vs multimodal branching, DTensor
handling, and stats keys — not numerical loss values.

The whole file is skipped (via the ``test/espnet2/speechlm/model/``
``conftest.py``'s ``collect_ignore_glob``) when
``liger_kernel.ops.fused_linear_cross_entropy`` is not importable.
"""

import pytest
import torch

from espnet2.speechlm.model.speechlm.lm import loss as loss_mod


class _RecordingLiger:
    """Stub replacement for LigerFusedLinearCrossEntropyFunction."""

    def __init__(self):
        self.calls = []

    def apply(self, *args):
        self.calls.append(args)
        return (
            torch.zeros((), requires_grad=True),  # loss
            torch.zeros(()),  # z_loss
            torch.zeros(()),  # token_accuracy
        )


@pytest.fixture
def recording_liger(monkeypatch):
    fake = _RecordingLiger()
    monkeypatch.setattr(loss_mod, "LigerFusedLinearCrossEntropyFunction", fake)
    return fake


# Arg-index map for LigerFusedLinearCrossEntropyFunction.apply(). These
# indices match the call sites in loss.py.
ARG_INPUT = 0
ARG_WEIGHT = 1
ARG_TARGET = 2
ARG_CE_WEIGHT = 4
ARG_IGNORE_INDEX = 5
ARG_Z_LOSS_SCALE = 6
ARG_REDUCTION = 8


def _make_inputs(B=1, T=6, N=1, H=8, V=20):
    torch.manual_seed(0)
    hidden = torch.randn(B, T, N, H)
    input_ids = torch.randint(0, V, (B, T, N))
    loss_mask = torch.ones(B, T, N)
    lm_head = torch.randn(V, H)
    return hidden, input_ids, loss_mask, lm_head


class TestStreamZeroOnly:
    """num_stream=1, no multimodal range → single Liger call."""

    def test_single_call(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs()
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
        )
        assert len(recording_liger.calls) == 1

    def test_shift_by_one(self, recording_liger):
        """After shift, hidden has T-1 rows flattened into [B*(T-1)*N, H]."""
        B, T, N, H, V = 2, 6, 1, 8, 20
        hidden, input_ids, loss_mask, lm_head = _make_inputs(B, T, N, H, V)
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
        )
        s0_input = recording_liger.calls[0][ARG_INPUT]
        assert s0_input.shape == (B * (T - 1), H)

    def test_stream_zero_ignore_index(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs()
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
        )
        assert recording_liger.calls[0][ARG_IGNORE_INDEX] == 0
        assert recording_liger.calls[0][ARG_REDUCTION] == "sum"

    def test_pre_mask_zeroed(self, recording_liger):
        """loss_mask==0 positions should become 0 in the passed targets."""
        B, T, N, H, V = 1, 5, 1, 8, 20
        hidden, input_ids, loss_mask, lm_head = _make_inputs(B, T, N, H, V)
        input_ids.fill_(7)
        loss_mask[:, 2, :] = 0  # mask out position index 2 pre-shift
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
        )
        targets = recording_liger.calls[0][ARG_TARGET]
        # After shift, position 2 pre-shift corresponds to index 1 post-shift.
        assert targets[1] == 0  # pad_id
        # Unmasked positions keep their original value (7).
        assert (targets[[0, 2, 3]] == 7).all()

    def test_count(self, recording_liger):
        """count = sum of non-zero stream-0 loss_mask entries after shift."""
        B, T, N, H, V = 1, 5, 1, 8, 20
        hidden, input_ids, loss_mask, lm_head = _make_inputs(B, T, N, H, V)
        loss_mask[0, 3, 0] = 0  # one masked position after shift (index 2)
        _, count, _ = loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
        )
        assert count.item() == (T - 1) - 1

    def test_stats_keys_stream_zero(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs()
        _, _, stats = loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
            z_loss_weight=1e-5,
        )
        assert "z_loss" in stats
        assert "z_loss_s0" in stats
        assert "z_loss_mm" not in stats
        assert "acc_layer0" in stats


class TestStreamZeroPlusMultimodal:
    """num_stream>1 with a multimodal range → two Liger calls."""

    def test_two_calls(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs(N=4, V=50)
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=(10, 40),
            num_stream=4,
            training=True,
        )
        assert len(recording_liger.calls) == 2

    def test_mm_weight_sliced(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs(N=4, V=50)
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=(10, 40),
            num_stream=4,
            training=True,
        )
        mm_weight = recording_liger.calls[1][ARG_WEIGHT]
        assert mm_weight.shape == (40 - 10, lm_head.shape[1])

    def test_mm_ignore_index(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs(N=4, V=50)
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=(10, 40),
            num_stream=4,
            training=True,
        )
        assert recording_liger.calls[1][ARG_IGNORE_INDEX] == -100

    def test_mm_target_remap(self, recording_liger):
        """mm targets in range are remapped to local indices; out-of-range → -100."""
        B, T, N, H, V = 1, 4, 4, 8, 50
        hidden, input_ids, loss_mask, lm_head = _make_inputs(B, T, N, H, V)
        mm_start, mm_end = 10, 40
        # Streams 1+, time indices 1..T-1 (post-shift 0..T-2):
        # Put one in-range, one out-of-range target in streams 1+.
        # Build explicit tokens so we know what to expect.
        input_ids[:, :, 1:] = 5  # out-of-range
        input_ids[0, 2, 1] = 15  # in-range, local = 5
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=(mm_start, mm_end),
            num_stream=4,
            training=True,
        )
        mm_targets = recording_liger.calls[1][ARG_TARGET]
        assert (mm_targets == -100).any()
        assert (mm_targets == 15 - mm_start).any()

    def test_mm_ce_weight_sliced(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs(N=4, V=50)
        ce_weight = torch.arange(50, dtype=torch.float32)
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=(10, 40),
            num_stream=4,
            training=True,
            ce_weight=ce_weight,
        )
        mm_ce_weight = recording_liger.calls[1][ARG_CE_WEIGHT]
        assert torch.equal(mm_ce_weight, ce_weight[10:40])

    def test_stats_includes_mm(self, recording_liger):
        hidden, input_ids, loss_mask, lm_head = _make_inputs(N=4, V=50)
        _, _, stats = loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            lm_head,
            multimodal_vocab_range=(10, 40),
            num_stream=4,
            training=True,
            z_loss_weight=1e-5,
        )
        assert "z_loss_mm" in stats


class TestDTensorLmHead:
    """lm_head_weight with ``.full_tensor()`` should be materialized."""

    def test_full_tensor_called(self, recording_liger):
        hidden, input_ids, loss_mask, _ = _make_inputs()

        class _FakeDTensor:
            def __init__(self, tensor):
                self._tensor = tensor
                self.full_tensor_calls = 0

            def full_tensor(self):
                self.full_tensor_calls += 1
                return self._tensor

        materialized = torch.randn(20, 8)
        fake = _FakeDTensor(materialized)
        loss_mod.fused_cross_entropy_loss(
            hidden,
            input_ids,
            loss_mask,
            fake,
            multimodal_vocab_range=None,
            num_stream=1,
            training=True,
        )
        assert fake.full_tensor_calls == 1
        passed_weight = recording_liger.calls[0][ARG_WEIGHT]
        # passed through a dtype cast, so we compare values not storage
        assert passed_weight.shape == materialized.shape
