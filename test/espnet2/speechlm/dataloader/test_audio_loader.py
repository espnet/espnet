"""Tests for espnet2/speechlm/dataloader/multimodal_loader/audio_loader.py.

All tests use pytest.importorskip — they skip in CI but run when
real dependencies (arkive, duckdb, lhotse, kaldiio) are installed.
"""

import pytest

arkive = pytest.importorskip("arkive", reason="arkive not installed")
duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")
pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")
lhotse = pytest.importorskip("lhotse", reason="lhotse not installed")


# ---------- KaldiAudioReader ----------


class TestKaldiAudioReader:
    @pytest.fixture(autouse=True)
    def _skip_without_kaldiio(self):
        pytest.importorskip("kaldiio", reason="kaldiio not installed")

    def test_kaldi_init(self, tmp_path):
        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            KaldiAudioReader,
        )

        index_file = tmp_path / "index.scp"
        index_file.write_text("utt1 /fake/ark:0\nutt2 /fake/ark:100\n")
        reader = KaldiAudioReader(str(index_file))
        assert len(reader) == 2
        assert set(reader.keys()) == {"utt1", "utt2"}

    def test_kaldi_valid_ids(self, tmp_path):
        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            KaldiAudioReader,
        )

        index_file = tmp_path / "index.scp"
        index_file.write_text("utt1 /fake/ark:0\nutt2 /fake/ark:100\n")
        reader = KaldiAudioReader(str(index_file), valid_ids=["utt1"])
        assert len(reader) == 1
        assert "utt1" in reader
        assert "utt2" not in reader

    def test_kaldi_contains(self, tmp_path):
        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            KaldiAudioReader,
        )

        index_file = tmp_path / "index.scp"
        index_file.write_text("utt1 /fake/ark:0\n")
        reader = KaldiAudioReader(str(index_file))
        assert "utt1" in reader
        assert "nonexistent" not in reader

    def test_kaldi_missing_key_raises(self, tmp_path):
        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            KaldiAudioReader,
        )

        index_file = tmp_path / "index.scp"
        index_file.write_text("utt1 /fake/ark:0\n")
        reader = KaldiAudioReader(str(index_file))
        with pytest.raises(KeyError, match="nonexistent"):
            reader["nonexistent"]

    def test_kaldi_getitem_shape(self, tmp_path):
        """Verify audio returned has shape [num_channels, num_samples]."""
        import numpy as np
        from unittest.mock import patch

        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            KaldiAudioReader,
        )

        index_file = tmp_path / "index.scp"
        index_file.write_text("utt1 /fake/ark:0\n")
        reader = KaldiAudioReader(str(index_file))

        # Mock kaldiio.load_mat to return (sample_rate, 1D_audio_array)
        mono_audio = np.zeros(16000, dtype=np.float32)
        with patch("kaldiio.load_mat", return_value=(16000, mono_audio)):
            audio, sr = reader["utt1"]
        assert sr == 16000
        assert audio.shape == (1, 16000)  # [num_channels, num_samples]


# ---------- LhotseAudioReader ----------


class TestLhotseAudioReader:
    def test_lhotse_no_manifest_raises(self, tmp_path):
        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            LhotseAudioReader,
        )

        with pytest.raises(FileNotFoundError, match="No manifest files found"):
            LhotseAudioReader(str(tmp_path))

    def test_lhotse_with_recordings(self, tmp_path):
        """Test loading from recordings manifest (mocked)."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            LhotseAudioReader,
        )

        # Create a fake recordings.jsonl.gz so the path check passes
        recordings_path = tmp_path / "recordings.jsonl.gz"
        recordings_path.touch()

        # Build mock recording objects
        mock_rec1 = MagicMock()
        mock_rec1.id = "rec1"
        mock_rec1.sampling_rate = 16000
        mock_rec1.load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_rec_set = MagicMock()
        mock_rec_set.__iter__ = MagicMock(return_value=iter([mock_rec1]))
        mock_rec_set.__contains__ = MagicMock(return_value=True)
        mock_rec_set.__len__ = MagicMock(return_value=1)
        mock_rec_set.__getitem__ = MagicMock(return_value=mock_rec1)

        with patch(
            "espnet2.speechlm.dataloader.multimodal_loader.audio_loader.RecordingSet"
        ) as MockRS:
            MockRS.from_file.return_value = mock_rec_set
            MockRS.from_recordings.return_value = mock_rec_set
            reader = LhotseAudioReader(str(tmp_path))

        audio, sr = reader["rec1"]
        assert sr == 16000
        assert audio.shape == (1, 16000)  # mono → [1, N]


# ---------- ArkiveAudioReader ----------


class TestArkiveAudioReader:
    def test_arkive_audio_smoke(self):
        """Minimal smoke test — only runs with full arkive/duckdb/pyarrow stack."""
        from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import (
            ArkiveAudioReader,
        )

        # Just verify the class is importable and callable
        assert callable(ArkiveAudioReader)
