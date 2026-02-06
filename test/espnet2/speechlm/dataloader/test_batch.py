"""Tests for espnet2/speechlm/dataloader/batch.py — batching algorithms."""

from unittest.mock import patch

import pytest

from espnet2.speechlm.dataloader.batch import (
    _bfd_worker,
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
        flat = [(l, k) for b in batches for l, k in b]
        assert sorted(flat) == sorted(items)
        for b in batches:
            assert sum(l for l, _ in b) <= 50

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
        flat = {(l, k) for b in batches for l, k in b}
        assert flat == set(items)
        for b in batches:
            assert sum(l for l, _ in b) <= 50

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
