"""Tests for espnet2/speechlm/dataloader/iterator.py — helpers & factory."""

import json

import pytest

from espnet2.speechlm.dataloader.iterator import (
    DataIteratorFactory,
    _load_stats,
    _parse_data_specifier,
    _resample,
)

# ---------- _parse_data_specifier ----------


class TestParseDataSpecifier:
    def test_parse_unreg_with_factor(self):
        unreg, reg = _parse_data_specifier("asr:libri:train.json:2.0", "")
        assert unreg == [("asr", "libri", "train.json", 2.0)]
        assert reg == []

    def test_parse_unreg_without_factor(self):
        unreg, reg = _parse_data_specifier("asr:libri:train.json", "")
        assert unreg == [("asr", "libri", "train.json", 1.0)]

    def test_parse_reg_with_factor(self):
        unreg, reg = _parse_data_specifier("", "tts:lj:1.5")
        assert unreg == []
        assert reg == [("tts", "lj", 1.5)]

    def test_parse_reg_without_factor(self):
        unreg, reg = _parse_data_specifier("", "tts:lj")
        assert reg == [("tts", "lj", 1.0)]

    def test_parse_multiple(self):
        unreg, reg = _parse_data_specifier(
            "asr:libri:a.json:2.0 tts:lj:b.json", "stt:cv:0.5"
        )
        assert len(unreg) == 2
        assert len(reg) == 1
        assert unreg[0] == ("asr", "libri", "a.json", 2.0)
        assert unreg[1] == ("tts", "lj", "b.json", 1.0)
        assert reg[0] == ("stt", "cv", 0.5)

    def test_parse_empty(self):
        unreg, reg = _parse_data_specifier("", "")
        assert unreg == []
        assert reg == []

    def test_parse_whitespace_only(self):
        unreg, reg = _parse_data_specifier("   ", "  ")
        assert unreg == []
        assert reg == []

    def test_parse_invalid_unreg(self):
        with pytest.raises(ValueError, match="Invalid unregistered specifier"):
            _parse_data_specifier("asr:libri", "")

    def test_parse_invalid_reg(self):
        with pytest.raises(ValueError, match="Invalid registered specifier"):
            _parse_data_specifier("", "onlyonepart")


# ---------- _load_stats ----------


class TestLoadStats:
    def test_load_stats_basic(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text('{"utt1": 100, "utt2": 200}\n')
        result = _load_stats(f, "asr", "libri")
        assert result == {
            ("asr", "libri", "utt1"): 100,
            ("asr", "libri", "utt2"): 200,
        }

    def test_load_stats_multiple_lines(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text('{"utt1": 100}\n{"utt2": 200}\n')
        result = _load_stats(f, "asr", "libri")
        assert len(result) == 2
        assert result[("asr", "libri", "utt1")] == 100
        assert result[("asr", "libri", "utt2")] == 200

    def test_load_stats_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text('\n{"utt1": 100}\n\n{"utt2": 200}\n\n')
        result = _load_stats(f, "asr", "libri")
        assert len(result) == 2

    def test_load_stats_file_not_found(self, tmp_path):
        f = tmp_path / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError):
            _load_stats(f, "asr", "libri")

    def test_load_stats_invalid_json(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text("not valid json\n")
        with pytest.raises(json.JSONDecodeError):
            _load_stats(f, "asr", "libri")

    def test_load_stats_non_dict_line(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text("[1, 2, 3]\n")
        with pytest.raises(ValueError, match="is not a dict"):
            _load_stats(f, "asr", "libri")

    def test_load_stats_non_numeric_length(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text('{"utt1": "abc"}\n')
        with pytest.raises(ValueError, match="Invalid length value"):
            _load_stats(f, "asr", "libri")

    def test_load_stats_float_to_int(self, tmp_path):
        f = tmp_path / "stats.jsonl"
        f.write_text('{"utt1": 100.7}\n')
        result = _load_stats(f, "asr", "libri")
        assert result[("asr", "libri", "utt1")] == 100
        assert isinstance(result[("asr", "libri", "utt1")], int)


# ---------- _resample ----------


class TestResample:
    def test_resample_factor_one(self):
        lst = [1, 2, 3]
        assert _resample(lst, 1.0) == [1, 2, 3]

    def test_resample_factor_two(self):
        lst = [1, 2, 3]
        assert _resample(lst, 2.0) == [1, 2, 3, 1, 2, 3]

    def test_resample_fractional(self):
        lst = [1, 2, 3, 4]
        result = _resample(lst, 1.5, seed=42)
        assert len(result) == 6  # int(1.5 * 4) = 4 + 2

    def test_resample_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _resample([1], 0)

    def test_resample_negative_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _resample([1], -1.0)

    def test_resample_empty_list(self):
        assert _resample([], 2.0) == []

    def test_resample_deterministic(self):
        lst = list(range(20))
        r1 = _resample(lst, 1.5, seed=99)
        r2 = _resample(lst, 1.5, seed=99)
        assert r1 == r2

    def test_resample_large_factor(self):
        result = _resample([1], 100.0)
        assert len(result) == 100
        assert all(x == 1 for x in result)


# ---------- DataIteratorFactory ----------


class TestDataIteratorFactory:
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create synthetic dataset JSON, text file, and stats JSONL."""
        # Text file
        text_file = tmp_path / "text.txt"
        text_file.write_text("utt1 hello world\nutt2 foo bar\nutt3 baz qux\n")

        # Dataset JSON
        dataset_json = tmp_path / "dataset.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "data_entry": [
                        {
                            "name": "text1",
                            "path": str(text_file),
                            "reader": "text",
                        }
                    ],
                    "samples": ["utt1", "utt2", "utt3"],
                }
            )
        )

        # Stats JSONL
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        stats_file = stats_dir / "stats_text_only_mydata.jsonl"
        stats_file.write_text('{"utt1": 100, "utt2": 200, "utt3": 150}\n')

        return {
            "dataset_json": str(dataset_json),
            "stats_dir": str(stats_dir),
            "text_file": str(text_file),
        }

    def test_factory_init_with_synthetic_data(self, synthetic_data):
        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
        )
        assert len(factory.batched_examples) > 0

    def test_build_index_no_shuffle(self, synthetic_data):
        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
            shuffle=False,
        )
        index = factory.build_index(seed=42)
        assert index == list(range(len(factory.batched_examples)))

    def test_build_index_shuffle(self, tmp_path):
        # Create dataset with many small items so bucket batching produces >1 batch
        samples = [f"utt{i}" for i in range(20)]
        text_file = tmp_path / "text.txt"
        text_file.write_text(
            "\n".join(f"{s} word{i}" for i, s in enumerate(samples)) + "\n"
        )
        dataset_json = tmp_path / "dataset.json"
        dataset_json.write_text(
            json.dumps(
                {
                    "data_entry": [
                        {"name": "text1", "path": str(text_file), "reader": "text"}
                    ],
                    "samples": samples,
                }
            )
        )
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        stats_file = stats_dir / "stats_text_only_mydata.jsonl"
        # Each item has length 10, batch_size=30 → 3 per batch → ~7 batches
        stats_file.write_text(json.dumps({s: 10 for s in samples}) + "\n")

        factory = DataIteratorFactory(
            unregistered_specifier=f"text_only:mydata:{dataset_json}",
            stats_dir=str(stats_dir),
            batch_size=30,
            num_workers=0,
            shuffle=True,
        )
        n = len(factory.batched_examples)
        assert n > 1, "Need multiple batches to test shuffling"
        idx1 = factory.build_index(seed=42)
        idx2 = factory.build_index(seed=42)
        assert idx1 == idx2  # deterministic with same seed
        assert sorted(idx1) == list(range(n))  # is a permutation

    def test_build_iter_returns_dataloader(self, synthetic_data):
        from torch.utils.data import DataLoader

        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
        )
        loader = factory.build_iter(global_step=0, length=1)
        assert isinstance(loader, DataLoader)

    def test_build_iter_negative_step_raises(self, synthetic_data):
        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
        )
        with pytest.raises(ValueError, match="non-negative"):
            factory.build_iter(global_step=-1, length=1)

    def test_build_iter_zero_length_raises(self, synthetic_data):
        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
        )
        with pytest.raises(ValueError, match="must be positive"):
            factory.build_iter(global_step=0, length=0)

    def test_build_iter_no_batches_raises(self, synthetic_data):
        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
        )
        factory.batched_examples = []
        with pytest.raises(ValueError, match="No batches available"):
            factory.build_iter(global_step=0, length=1)

    def test_save_load_state(self, synthetic_data, tmp_path):
        factory = DataIteratorFactory(
            unregistered_specifier=(
                f"text_only:mydata:{synthetic_data['dataset_json']}"
            ),
            stats_dir=synthetic_data["stats_dir"],
            batch_size=10000,
            num_workers=0,
        )
        state_file = tmp_path / "state.json"
        factory.save_iterator_state(str(state_file))
        loaded = factory.load_iterator_state(str(state_file))
        # Convert original to tuples for comparison (JSON round-trips lose tuples)
        expected = [[tuple(ex) for ex in batch] for batch in factory.batched_examples]
        assert loaded == expected

    def test_load_state_missing_key(self, tmp_path):
        state_file = tmp_path / "bad_state.json"
        state_file.write_text('{"wrong_key": []}')
        factory = DataIteratorFactory.__new__(DataIteratorFactory)
        with pytest.raises(KeyError, match="batched_examples"):
            factory.load_iterator_state(str(state_file))
