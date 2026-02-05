from pathlib import Path
from unittest import mock

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


def test_download_url_accepts_none_logger(monkeypatch, tmp_path: Path, capsys):
    def fake_urlretrieve(url, filename, reporthook):
        reporthook(1, 1, 1)

    monkeypatch.setattr(download_utils.urllib.request, "urlretrieve", fake_urlretrieve)
    download_utils.download_url(
        "http://example.com/file", tmp_path / "file", logger=None
    )
    out = capsys.readouterr().out
    assert "Start download" in out
    assert "Download completed" in out


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

        def extractall(self, path):
            opened["path"] = path

    def fake_open(path, mode):
        opened["path_arg"] = path
        opened["mode"] = mode
        return DummyTar()

    monkeypatch.setattr(download_utils.tarfile, "open", fake_open)
    logger = mock.Mock()

    download_utils.extract_targz(archive, tmp_path / "dst", logger)

    assert opened["mode"] == "r:gz"
    assert opened["path"] == tmp_path / "dst"


def test_extract_targz_accepts_none_logger(monkeypatch, tmp_path: Path, capsys):
    archive = tmp_path / "dummy.tar.gz"

    class DummyTar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def extractall(self, path):
            pass

    monkeypatch.setattr(
        download_utils.tarfile, "open", lambda *_args, **_kwargs: DummyTar()
    )
    download_utils.extract_targz(archive, tmp_path, logger=None)
    out = capsys.readouterr().out
    assert "Extracting" in out
