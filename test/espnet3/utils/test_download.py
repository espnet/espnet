from pathlib import Path
from unittest import mock

import pytest

from espnet3.utils import download_utils


def test_setup_logger_returns_logger(tmp_path: Path):
    logger = download_utils.setup_logger("test_logger", log_dir=tmp_path)
    logger.info("hello")
    # Should create a log file
    assert (tmp_path / "download.log").exists()


def test_download_progress_logs_buckets():
    logger = mock.Mock()
    progress = download_utils.DownloadProgress(
        logger=logger, name="file", step_percent=10
    )

    # Simulate increasing blocks; ensure logger.info is called multiple times
    progress(block_num=1, block_size=10, total_size=100)
    progress(block_num=5, block_size=10, total_size=100)
    progress(block_num=9, block_size=10, total_size=100)

    assert logger.info.call_count >= 2


def test_download_url_invokes_urlretrieve(monkeypatch, tmp_path: Path):
    called = {}

    def fake_urlretrieve(url, filename, reporthook):
        called["url"] = url
        called["filename"] = filename
        # call hook once to simulate progress
        reporthook(1, 1, 1)

    monkeypatch.setattr(download_utils.urllib.request, "urlretrieve", fake_urlretrieve)
    logger = mock.Mock()

    download_utils.download_url("http://example.com/file", tmp_path / "file", logger)

    assert called["url"] == "http://example.com/file"
    assert Path(called["filename"]) == tmp_path / "file"
    assert logger.info.call_count >= 2  # start and completed


def test_download_url_accepts_none_logger(monkeypatch, tmp_path: Path, caplog):
    def fake_urlretrieve(url, filename, reporthook):
        reporthook(1, 1, 1)

    monkeypatch.setattr(download_utils.urllib.request, "urlretrieve", fake_urlretrieve)
    with caplog.at_level("INFO"):
        download_utils.download_url(
            "http://example.com/file", tmp_path / "file", logger=None
        )
    assert "Start download" in caplog.text
    assert "Download completed" in caplog.text


def test_extract_targz(monkeypatch, tmp_path: Path):
    archive = tmp_path / "dummy.tar.gz"
    (tmp_path / "dst").mkdir()
    opened = {}

    class DummyTar:
        def __enter__(self):
            opened["enter"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            opened["exit"] = True

        def extractall(self, path, filter=None):
            opened["path"] = path
            opened["filter"] = filter

    def fake_open(path, mode):
        opened["path_arg"] = path
        opened["mode"] = mode
        return DummyTar()

    monkeypatch.setattr(download_utils.tarfile, "open", fake_open)
    logger = mock.Mock()

    download_utils.extract_targz(archive, tmp_path / "dst", logger)

    assert opened["mode"] == "r:gz"
    assert opened["path"] == tmp_path / "dst"
    # Anything but "data" lets a crafted archive write outside dst.
    assert opened["filter"] == "data"


def test_extract_targz_accepts_none_logger(monkeypatch, tmp_path: Path, caplog):
    archive = tmp_path / "dummy.tar.gz"

    class DummyTar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def extractall(self, path, filter=None):
            pass

    monkeypatch.setattr(
        download_utils.tarfile, "open", lambda *_args, **_kwargs: DummyTar()
    )
    with caplog.at_level("INFO"):
        download_utils.extract_targz(archive, tmp_path, logger=None)
    assert "Extracting" in caplog.text


def test_extract_targz_refuses_traversal(tmp_path: Path):
    """A member escaping dst_dir must not be written.

    The other extract_targz tests stub tarfile out entirely, so this is the
    only one that drives real extraction.
    """
    import io
    import tarfile

    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        payload = b"PWNED"
        info = tarfile.TarInfo("../../escaped.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    dst = tmp_path / "deep" / "dst"
    dst.mkdir(parents=True)

    with pytest.raises(tarfile.TarError):
        download_utils.extract_targz(archive, dst, logger=None)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "deep" / "escaped.txt").exists()
