"""Exercise the official ZIP preparation path with a local HTTP server."""

import io
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from egs3.voxlingua107.esp2_lid.dataset import builder as builder_module
from espnet3.systems.base import system as system_module
from espnet3.systems.esp2_lid.system import LIDSystem


@pytest.fixture
def official_source(tmp_path, monkeypatch):
    monkeypatch.setattr(builder_module, "_ISO3_CODES", {"aa": "aaa", "bb": "bbb"})
    audio = io.BytesIO()
    sf.write(audio, np.arange(160, dtype=np.float32) / 160, 16000, format="WAV")
    wav = audio.getvalue()
    files = {}
    for name, members in {
        "aa.zip": ["aa/train.wav"],
        "bb.zip": ["bb/train.wav"],
        "dev.zip": ["aa/dev.wav"],
    }.items():
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            for member in members:
                z.writestr(member, wav)
        files["/" + name] = archive.getvalue()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Range")))
            payload = files.get(self.path)
            if payload is None:
                self.send_error(404)
                return
            start = 0
            if self.headers.get("Range"):
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
            if start >= len(payload):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.end_headers()
                return
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            if start:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    files["/zip_urls.txt"] = f"{url}/aa.zip\n{url}/bb.zip\n".encode()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            root=tmp_path / "source",
            files=files,
            requests=requests,
            wav=wav,
            kwargs={
                "source_dir": tmp_path / "source",
                "zip_urls_url": url + "/zip_urls.txt",
                "dev_zip_url": url + "/dev.zip",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_create_dataset_downloads_extracts_and_reuses_source(
    official_source, tmp_path, monkeypatch
):
    source = official_source
    monkeypatch.setattr(
        system_module,
        "load_dataset_module",
        lambda **kwargs: SimpleNamespace(
            DatasetBuilder=builder_module.VoxLingua107Builder
        ),
    )
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": {"train": [{}], "valid": [{}]},
            "create_dataset": {
                **{k: str(v) for k, v in source.kwargs.items()},
                "data_dir": str(tmp_path / "metadata"),
            },
        }
    )
    system = LIDSystem(training_config=config)
    system.create_dataset()
    assert (source.root / "aa/train.wav").read_bytes() == source.wav
    assert (source.root / "bb/train.wav").read_bytes() == source.wav
    assert (source.root / "dev/aa/dev.wav").read_bytes() == source.wav
    assert (tmp_path / "metadata/train/category2utt").read_text() == "aaa 0\nbbb 1\n"
    assert (tmp_path / "metadata/dev/category2utt").read_text() == "aaa 0\n"
    before = len(source.requests)
    system.create_dataset()
    assert len(source.requests) == before
    assert not (source.root / builder_module._PREPARING).exists()


def test_resumes_partial_zip_preserves_complete_zip(official_source):
    source = official_source
    source.root.mkdir()
    complete = source.root / "aa.zip"
    complete.write_bytes(source.files["/aa.zip"])
    timestamp = complete.stat().st_mtime_ns
    (source.root / "bb.zip").write_bytes(source.files["/bb.zip"][:64])
    builder_module.VoxLingua107Builder().prepare_source(**source.kwargs)
    assert ("/bb.zip", "bytes=64-") in source.requests
    assert complete.stat().st_mtime_ns == timestamp
    for name in ("aa.zip", "bb.zip", "dev.zip"):
        assert (source.root / name).read_bytes() == source.files["/" + name]


def test_interrupted_extraction_is_not_treated_as_complete(
    official_source, monkeypatch
):
    source = official_source
    builder = builder_module.VoxLingua107Builder()
    run = subprocess.run

    def interrupt_after_dev_extraction(args, **kwargs):
        result = run(args, **kwargs)
        if args[0] == "unzip" and args[3].endswith("dev.zip"):
            raise subprocess.CalledProcessError(1, args)
        return result

    with monkeypatch.context() as m:
        m.setattr(builder_module.subprocess, "run", interrupt_after_dev_extraction)
        with pytest.raises(subprocess.CalledProcessError):
            builder.prepare_source(**source.kwargs)
    assert (source.root / "dev/aa/dev.wav").is_file()
    assert not builder.is_source_prepared(source_dir=source.root)
    builder.prepare_source(**source.kwargs)
    assert builder.is_source_prepared(source_dir=source.root)
