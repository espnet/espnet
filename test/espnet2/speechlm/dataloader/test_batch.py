"""Tests for espnet2/speechlm/dataloader/batch.py — batching algorithms."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from espnet2.speechlm.dataloader.batch import (
    _bfd_worker,
    _current_accelerator_device,
    _diverse_bfd_worker,
    batchfy,
    batchfy_bucket,
    batchfy_pack,
    synchronize_batches,
)

# ---------- batchfy_bucket ----------


class TestBatchfyBucket:
    def test_bucket_basic(self):
        keys = ["a", "b", "c", "d"]
        key_to_length = {"a": 10, "b": 20, "c": 30, "d": 40}
        batches = batchfy_bucket(keys, key_to_length, batch_token=60)
        # All keys must appear
        flat = [k for b in batches for k in b]
        assert sorted(flat) == sorted(keys)
        # Each batch must respect the token limit: max_len * size <= token
        for b in batches:
            max_len = max(key_to_length[k] for k in b)
            assert max_len * len(b) <= 60

    def test_bucket_empty(self):
        assert batchfy_bucket([], {}, batch_token=100) == []

    def test_bucket_single_item(self):
        batches = batchfy_bucket(["x"], {"x": 50}, batch_token=100)
        assert batches == [["x"]]

    def test_bucket_same_length(self):
        keys = [f"k{i}" for i in range(10)]
        key_to_length = {k: 10 for k in keys}
        batches = batchfy_bucket(keys, key_to_length, batch_token=30)
        for b in batches:
            assert len(b) <= 3  # 10 * 3 = 30
        flat = [k for b in batches for k in b]
        assert sorted(flat) == sorted(keys)

    def test_bucket_large_token(self):
        keys = ["a", "b", "c"]
        key_to_length = {"a": 1, "b": 2, "c": 3}
        batches = batchfy_bucket(keys, key_to_length, batch_token=10000)
        assert len(batches) == 1
        assert sorted(batches[0]) == sorted(keys)

    def test_bucket_each_own_batch(self):
        keys = ["a", "b", "c"]
        key_to_length = {"a": 100, "b": 100, "c": 100}
        batches = batchfy_bucket(keys, key_to_length, batch_token=100)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 1


# ---------- _bfd_worker ----------


class TestBfdWorker:
    def test_bfd_worker_basic(self):
        items = [(10, "a"), (20, "b"), (30, "c"), (15, "d")]
        batches = _bfd_worker(items, batch_token=50)
        flat = [(ln, k) for b in batches for ln, k in b]
        assert sorted(flat) == sorted(items)
        for b in batches:
            assert sum(ln for ln, _ in b) <= 50

    def test_bfd_worker_packing(self):
        # 30 + 20 = 50, should fit in one batch
        items = [(30, "a"), (20, "b")]
        batches = _bfd_worker(items, batch_token=50)
        assert len(batches) == 1

    def test_bfd_worker_empty(self):
        assert _bfd_worker([], batch_token=100) == []

    def test_bfd_worker_single(self):
        items = [(10, "a")]
        batches = _bfd_worker(items, batch_token=100)
        assert len(batches) == 1


# ---------- _diverse_bfd_worker ----------


class TestDiverseBfdWorker:
    def test_diverse_bfd_empty(self):
        assert _diverse_bfd_worker([], batch_token=100) == []

    def test_diverse_bfd_basic(self):
        items = [(10, "a"), (20, "b"), (30, "c"), (15, "d"), (25, "e")]
        batches = _diverse_bfd_worker(items, batch_token=50)
        flat = {(ln, k) for b in batches for ln, k in b}
        assert flat == set(items)
        for b in batches:
            assert sum(ln for ln, _ in b) <= 50

    def test_diverse_bfd_deterministic(self):
        items = [(i, f"k{i}") for i in range(1, 21)]
        b1 = _diverse_bfd_worker(items, batch_token=50)
        b2 = _diverse_bfd_worker(items, batch_token=50)
        assert b1 == b2


# ---------- batchfy_pack ----------


class TestBatchfyPack:
    def test_pack_small_input(self):
        keys = ["a", "b", "c"]
        key_to_length = {"a": 10, "b": 20, "c": 15}
        batches = batchfy_pack(keys, key_to_length, batch_token=50)
        flat = [k for b in batches for k in b]
        assert sorted(flat) == sorted(keys)

    def test_pack_returns_keys_only(self):
        keys = ["a", "b"]
        key_to_length = {"a": 10, "b": 20}
        batches = batchfy_pack(keys, key_to_length, batch_token=50)
        for b in batches:
            for item in b:
                assert isinstance(item, str)


# ---------- batchfy (dispatcher) ----------


class TestBatchfy:
    def test_batchfy_bucket_method(self):
        keys = ["a", "b", "c"]
        key_to_length = {"a": 10, "b": 20, "c": 30}
        with patch(
            "espnet2.speechlm.dataloader.batch.synchronize_batches",
            side_effect=lambda x: x,
        ):
            batches = batchfy(keys, key_to_length, 60, "bucket")
        flat = [k for b in batches for k in b]
        assert sorted(flat) == sorted(keys)

    def test_batchfy_pack_method(self):
        keys = ["a", "b"]
        key_to_length = {"a": 10, "b": 20}
        with patch(
            "espnet2.speechlm.dataloader.batch.synchronize_batches",
            side_effect=lambda x: x,
        ):
            batches = batchfy(keys, key_to_length, 50, "pack")
        flat = [k for b in batches for k in b]
        assert sorted(flat) == sorted(keys)

    def test_batchfy_invalid_method(self):
        with patch(
            "espnet2.speechlm.dataloader.batch.synchronize_batches",
            side_effect=lambda x: x,
        ):
            with pytest.raises(ValueError, match="Invalid batch_method"):
                batchfy(["a"], {"a": 10}, 100, "invalid")

    def test_batchfy_discards_oversized(self, caplog):
        import logging

        keys = ["small", "big"]
        key_to_length = {"small": 10, "big": 200}
        with caplog.at_level(
            logging.WARNING, logger="espnet2.speechlm.dataloader.batch"
        ):
            with patch(
                "espnet2.speechlm.dataloader.batch.synchronize_batches",
                side_effect=lambda x: x,
            ):
                batches = batchfy(keys, key_to_length, 50, "bucket")
        flat = [k for b in batches for k in b]
        assert "small" in flat
        assert "big" not in flat
        assert any("Discarded 1 samples" in msg for msg in caplog.messages)

    def test_batchfy_all_oversized(self):
        keys = ["a", "b"]
        key_to_length = {"a": 200, "b": 300}
        with patch(
            "espnet2.speechlm.dataloader.batch.synchronize_batches",
            side_effect=lambda x: x,
        ):
            batches = batchfy(keys, key_to_length, 50, "bucket")
        assert batches == []


# ---------- synchronize_batches ----------


class TestSynchronizeBatches:
    def test_sync_no_cuda(self):
        batches = [["a", "b"], ["c"]]
        with patch("torch.cuda.is_available", return_value=False):
            result = synchronize_batches(batches)
        assert result == batches

    def test_sync_not_initialized(self):
        batches = [["a", "b"], ["c"]]
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            result = synchronize_batches(batches)
        assert result == batches

    def test_sync_initialized_but_no_accelerator(self):
        # torch.distributed is initialized but there is no accelerator to run the
        # collective on: synchronizing is impossible, so this must not be a silent
        # no-op (ranks would keep different numbers of batches).
        batches = [["a", "b"], ["c"]]
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "espnet2.speechlm.dataloader.batch._current_accelerator_device",
                return_value=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="requires an accelerator"):
                synchronize_batches(batches)

    def test_sync_pads_shorter_ranks(self):
        # A rank with fewer batches than the max across ranks is padded from the end.
        batches = [["a", "b"], ["c"]]

        def fake_all_gather(out_list, _tensor):
            for t in out_list:
                t.fill_(3)  # pretend some rank reports 3 batches

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "espnet2.speechlm.dataloader.batch._current_accelerator_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "espnet2.speechlm.dataloader.batch.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "espnet2.speechlm.dataloader.batch.dist.all_gather",
                side_effect=fake_all_gather,
            ),
        ):
            result = synchronize_batches(batches)
        assert len(result) == 3
        assert result[-1] == ["c"]


# ---------- _current_accelerator_device ----------


class TestCurrentAcceleratorDevice:
    def test_accelerator_available(self, monkeypatch):
        # torch >= 2.5 with an available accelerator: report its device.
        fake = SimpleNamespace(
            is_available=lambda: True,
            current_accelerator=lambda: torch.device("cpu"),
        )
        monkeypatch.setattr(torch, "accelerator", fake, raising=False)
        assert _current_accelerator_device() == torch.device("cpu")

    def test_accelerator_present_but_unavailable(self, monkeypatch):
        # torch.accelerator exists but reports nothing usable -> None.
        fake = SimpleNamespace(is_available=lambda: False)
        monkeypatch.setattr(torch, "accelerator", fake, raising=False)
        assert _current_accelerator_device() is None

    def test_no_accelerator_falls_back_to_cuda(self, monkeypatch):
        # torch < 2.5 (no torch.accelerator) with CUDA available -> "cuda".
        monkeypatch.delattr(torch, "accelerator", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _current_accelerator_device() == torch.device("cuda")

    def test_no_accelerator_no_cuda(self, monkeypatch):
        # torch < 2.5 and no CUDA -> None.
        monkeypatch.delattr(torch, "accelerator", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _current_accelerator_device() is None
