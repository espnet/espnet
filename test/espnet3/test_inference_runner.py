# tests/test_stft_runner_provider.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

# ==== SUT imports ====
from espnet3.runner.base_runner import BaseRunner
from espnet3.runner.inference_provider import InferenceProvider


class STFTProvider(InferenceProvider):
    @staticmethod
    def build_dataset(cfg: DictConfig):
        dataset = []
        for i, p in enumerate(cfg.dataset.audio_path):
            x, sr = sf.read(p, dtype="float32")
            dataset.append({"utt_id": f"utt{i}", "audio": x, "sr": int(sr)})
        return dataset

    @staticmethod
    def build_model(cfg: DictConfig):
        return {}


class STFTRunner(BaseRunner):
    @staticmethod
    def _stft_with_buffer(
        audio_chunk: np.ndarray,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        if "buffer" not in state:
            state["buffer"] = torch.zeros(n_fft - hop_length)
        if "window" not in state:
            state["window"] = torch.hann_window(win_length)

        buf: torch.Tensor = state["buffer"]
        win: torch.Tensor = state["window"]

        chunk_tensor = torch.from_numpy(audio_chunk).float()
        audio = torch.cat([buf, chunk_tensor], dim=0)

        if audio.shape[0] < n_fft:
            state["buffer"] = audio
            return {"stft": {"type": "text", "value": "[]"}}

        num_frames = (audio.shape[0] - n_fft) // hop_length + 1
        if num_frames <= 0:
            state["buffer"] = audio
            return {"stft": {"type": "text", "value": "[]"}}

        stft = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            center=False,
            return_complex=True,
        )
        next_frame_start = hop_length * num_frames
        state["buffer"] = audio[next_frame_start:]

        return {"stft": {"type": "text", "value": str(list(stft.shape))}}

    @staticmethod
    def _chunk_audio(x: np.ndarray, sr: int, chunk_sec: float) -> List[np.ndarray]:
        cs = max(1, int(sr * chunk_sec))
        return [x[i : i + cs] for i in range(0, len(x), cs)]

    @staticmethod
    def forward(idx: int, *, dataset, model, **env) -> Dict[str, Any]:
        sample = dataset[idx]
        x: np.ndarray = sample["audio"]
        sr: int = int(sample.get("sr", 16000))

        stream = bool(env.get("stream", False))
        n_fft = int(env.get("n_fft", 512))
        hop_length = int(env.get("hop_length", 128))
        win_length = int(env.get("win_length", 512))
        chunk_sec = sample.get("chunk_sec", env.get("chunk_sec", None))

        state: Dict[str, torch.Tensor] = {}

        if not stream:
            return STFTRunner._stft_with_buffer(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                state=state,
            )

        assert chunk_sec is not None, "Set chunk_sec when stream=True"
        chunks = STFTRunner._chunk_audio(x, sr, float(chunk_sec))
        outputs = [
            STFTRunner._stft_with_buffer(
                ch,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                state=state,
            )
            for ch in chunks
        ]

        return outputs[-1] if outputs else {"stft": {"type": "text", "value": "[]"}}


def write_output(utt_id: str, output: dict, out_dir: Path):
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    for key, spec in output.items():
        scp = out_dir / f"{key}.scp"
        scp.touch(exist_ok=True)

        typ = spec["type"]
        val = spec["value"]

        if typ == "text":
            # scp: "utt value"
            with scp.open("a", encoding="utf-8") as w:
                w.write(f"{utt_id} {val}\n")

        elif typ == "audio":
            wav_dir = out_dir / "data" / key
            wav_dir.mkdir(parents=True, exist_ok=True)
            path = wav_dir / f"{utt_id}.flac"
            sf.write(path.as_posix(), val.astype(np.float32), 16000, format="FLAC")
            with scp.open("a", encoding="utf-8") as w:
                w.write(f"{utt_id} {path.as_posix()}\n")

        else:
            raise ValueError(f"Unknown type: {typ}")


# ===============================================================
# Fixtures
# ===============================================================
@pytest.fixture(scope="module")
def test_audio_paths():
    base = Path("test_utils/espnet3/audio")
    assert base.exists(), f"Test audio directory not found: {base}"
    paths = sorted(base.glob("*.wav"))
    assert len(paths) > 0, f"No .wav files found in {base}"
    return paths


def _make_cfg_from_samples(
    audio_path: List[str], *, stream=False, chunk_sec: float | None = None
):
    ds = {"audio_path": audio_path}
    md = {"stream": stream, "n_fft": 512, "hop_length": 128, "win_length": 512}
    cfg = OmegaConf.create({"dataset": ds, "model": md})
    params = {}
    if chunk_sec is not None:
        params["chunk_sec"] = float(chunk_sec)
    params["stream"] = bool(stream)
    return cfg, params


def test_offline_on_example(test_audio_paths):
    cfg, params = _make_cfg_from_samples(test_audio_paths, stream=False)
    provider = STFTProvider(cfg, params=params)
    runner = STFTRunner(provider)

    out = runner([0])[0]
    assert "stft" in out and isinstance(out["stft"]["value"], str)
    assert "[]" not in out["stft"]["value"]


def test_write_output(tmp_path):
    output = {"stft": {"type": "text", "value": "test"}}
    write_output("utt1", output, tmp_path)
    scp_path = Path(tmp_path) / "stft.scp"
    assert scp_path.exists()
    content = scp_path.read_text().strip()
    assert content == "utt1 test"


def test_image_output_type_unsupported(tmp_path):
    img = (np.zeros((10, 10)) + 255 * np.tri(10, 10)).astype(np.uint8)
    image_output = {"img": {"type": "image", "value": img}}

    with pytest.raises(ValueError, match=r"Unknown type: image"):
        write_output("image_id1", image_output, tmp_path)


def test_audio_output_type(tmp_path):
    audio_output = {"speech": {"type": "audio", "value": np.random.random(16000)}}
    write_output("speech_id1", audio_output, tmp_path)
    assert (tmp_path / "speech.scp").exists()
    assert (tmp_path / "data" / "speech" / "speech_id1.flac").exists()


def test_streaming_on_example(test_audio_paths):
    cfg, params = _make_cfg_from_samples(test_audio_paths, stream=True, chunk_sec=0.1)

    provider = STFTProvider(cfg, params=params)
    runner = STFTRunner(provider)

    out = runner([0])[0]
    assert "stft" in out and isinstance(out["stft"]["value"], str)
    assert "[]" not in out["stft"]["value"]


def test_streaming_chunked_processing_manual(test_audio_paths):
    path = test_audio_paths[0]
    x, sr = sf.read(path, dtype="float32")

    chunk_sec = 0.1
    cs = int(sr * chunk_sec)
    chunks = [x[i : i + cs] for i in range(0, len(x), cs)]

    state = {}

    outs = [
        STFTRunner._stft_with_buffer(
            ch,
            n_fft=512,
            hop_length=128,
            win_length=512,
            state=state,
        )
        for ch in chunks
    ]
    result = outs[-1] if outs else {"stft": {"type": "text", "value": "[]"}}

    assert "stft" in result
    assert isinstance(result["stft"]["value"], str)
    assert "[]" not in result["stft"]["value"]


@pytest.mark.parametrize(
    "chunk_sec, expected_min_chunks", [(0.01, 90), (0.1, 9), (0.5, 2)]
)
def test_streaming_chunk_count(test_audio_paths, chunk_sec, expected_min_chunks):
    path = test_audio_paths[0]
    x, sr = sf.read(path, dtype="float32")
    chunks = STFTRunner._chunk_audio(x, sr, chunk_sec)
    assert isinstance(chunks, list)
    assert len(chunks) >= expected_min_chunks
    assert all(isinstance(c, np.ndarray) for c in chunks)


def _read_jsonl_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class _DummyFuture:
    def __init__(self, result=None):
        self._result = result


def _make_fake_client(num_workers: int = 2):
    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def scheduler_info(self):
            return {"workers": {f"w{i}": {} for i in range(num_workers)}}

        def submit(self, fn, *args, **kwargs):
            result = fn(*args, **kwargs)
            return _DummyFuture(result=result)

        def gather(self, futures):
            return [getattr(f, "_result", None) for f in futures]

    return _FakeClient()


def _patch_parallel_client_no_submit(monkeypatch, n_workers: int = 2):
    """Patch client.submit/map to Run Locally and Synchronously

    - Change client.submit/map to fully local, immediate (synchronous) execution
    - Make sure to also patch any symbols already imported inside base_runner
    - For async submission, skip actual job submission and directly call
        _async_worker_entry_from_spec_path locally
    - Override _run_async to return None so it matches existing tests expecting
        ret is None
    """
    from omegaconf import OmegaConf

    import espnet3.parallel.parallel as par
    import espnet3.runner.base_runner as br

    class _FakeCluster:
        def __init__(self):
            self.job_cls = None

        def job_script(self):
            return "#!/bin/sh\necho OK\n"

    class _FakeClient:
        def __init__(self):
            self.cluster = _FakeCluster()

        def close(self):
            pass

        def scheduler_info(self):
            return {
                "address": "fake://scheduler",
                "workers": {f"w{i}": {} for i in range(n_workers)},
            }

    def _fake_get_parallel_config():
        return OmegaConf.create({"env": "slurm", "n_workers": n_workers, "options": {}})

    def _fake_make_client(_cfg=None):
        return _FakeClient()

    monkeypatch.setattr(
        par, "get_parallel_config", _fake_get_parallel_config, raising=True
    )
    monkeypatch.setattr(par, "make_client", _fake_make_client, raising=True)

    monkeypatch.setattr(
        br, "get_parallel_config", _fake_get_parallel_config, raising=True
    )
    monkeypatch.setattr(br, "make_client", _fake_make_client, raising=True)

    def _fake_get_job_cls(cluster, spec_path=None):
        class _Job:
            @classmethod
            async def _submit_job(cls, _tf, *args, **kwargs):
                br._async_worker_entry_from_spec_path(spec_path)
                return "OK"

        return _Job

    monkeypatch.setattr(br, "get_job_cls", _fake_get_job_cls, raising=True)

    orig_run_async = br.BaseRunner._run_async

    async def _fake_run_async_return_none(self, indices):
        await orig_run_async(self, indices)
        return None

    monkeypatch.setattr(
        br.BaseRunner, "_run_async", _fake_run_async_return_none, raising=True
    )


def _collect_jsonl_rows(result_dir: Path):
    files = sorted(result_dir.glob("result-*.jsonl"))
    all_rows = []
    for fp in files:
        all_rows.extend(list(_read_jsonl_file(fp)))
    return files, all_rows


def _assert_stft_json(obj: dict):
    assert isinstance(obj, dict)
    assert "stft" in obj and isinstance(obj["stft"], dict)
    assert obj["stft"].get("type") == "text"
    val = obj["stft"].get("value")
    assert isinstance(val, str)
    assert "[]" not in val


def test_async_jsonl_offline_with_base_runner(test_audio_paths, tmp_path, monkeypatch):
    _patch_parallel_client_no_submit(monkeypatch, n_workers=2)

    cfg, params = _make_cfg_from_samples(test_audio_paths, stream=False)
    provider = STFTProvider(cfg, params=params)
    runner = STFTRunner(
        provider,
        async_mode=True,
        async_specs_dir=tmp_path / "_specs",
        async_result_dir=tmp_path / "_results_offline",
    )

    indices = list(range(min(4, len(test_audio_paths))))
    ret = runner(indices)

    assert ret is None

    result_dir = tmp_path / "_results_offline"
    files, rows = _collect_jsonl_rows(result_dir)
    assert len(files) >= 1
    assert len(rows) == len(indices)

    for obj in rows:
        _assert_stft_json(obj)


def test_async_jsonl_streaming_with_base_runner(
    test_audio_paths, tmp_path, monkeypatch
):
    _patch_parallel_client_no_submit(monkeypatch, n_workers=3)

    cfg, params = _make_cfg_from_samples(test_audio_paths, stream=True, chunk_sec=0.1)
    provider = STFTProvider(cfg, params=params)
    runner = STFTRunner(
        provider,
        async_mode=True,
        async_specs_dir=tmp_path / "_specs",
        async_result_dir=tmp_path / "_results_stream",
    )

    indices = list(range(min(5, len(test_audio_paths))))
    ret = runner(indices)
    assert ret is None

    result_dir = tmp_path / "_results_stream"
    files, rows = _collect_jsonl_rows(result_dir)
    assert len(files) >= 1
    assert len(rows) == len(indices)

    for obj in rows:
        _assert_stft_json(obj)
