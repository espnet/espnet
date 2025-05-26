# tests/test_inference_runner.py
import shutil
from pathlib import Path

import pytest
import torch
# from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from espnet3.inference_runner import InferenceRunner
from espnet3.metrics import AbsMetrics


def load_line(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


class DummyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"text": f"hello {idx}"}


class IdentityTransform:
    def __call__(self, x):
        return x


class DummyInference(torch.nn.Module):
    def __call__(self, batch):
        return {
            "text": {"type": "text", "value": batch["text"]},
            "hypothesis": {"type": "text", "value": batch["text"].upper()},
        }


class DummyMetrics(AbsMetrics):
    def __call__(self, decode_dir, test_name, inputs):
        (decode_dir / test_name / "dummy_score").write_text("ok")
        return {"dummy": 42}


@pytest.fixture
def config():
    path = Path("test_utils/espnet3_dummy/inference_runner_config.yaml")
    OmegaConf.register_new_resolver("load_line", load_line)
    cfg = OmegaConf.load(path)
    return cfg


# === Test Fixtures ===
@pytest.fixture(scope="module")
def test_audio_paths():
    base = Path("test_utils/utils3/audio")
    assert base.exists(), f"Test audio directory not found: {base}"
    paths = sorted(base.glob("*.wav"))
    assert len(paths) > 0, f"No .wav files found in {base}"
    return paths


# === Offline inference ===
def test_A1_offline_on_example(test_audio_paths):
    path = test_audio_paths[0]
    x, _ = sf.read(path, dtype="float32")
    runner = STFTInferenceRunner(stream=False)
    sample = {"audio_path": str(path), "audio": x}
    out = runner.run_on_example("utt1", sample)
    assert "stft" in out and isinstance(out["stft"]["value"], str)
    assert "[]" not in out["stft"]["value"]


@pytest.mark.parametrize("path_idx", [0, 1])
def test_A3_read_audio_offline(test_audio_paths, path_idx):
    if path_idx >= len(test_audio_paths):
        pytest.skip(f"Not enough test files: {len(test_audio_paths)} < {path_idx + 1}")
    path = str(test_audio_paths[path_idx])
    runner = STFTInferenceRunner()
    wav = runner.read("audio", path, stream=False)
    print(wav)
    assert isinstance(wav, np.ndarray)
    assert wav.size > 0, "Audio file is empty"


def test_A4_read_text_offline(tmp_path):
    file_path = tmp_path / "text.txt"
    file_path.write_text("hello world")
    runner = STFTInferenceRunner()
    text = runner.read("text", str(file_path), stream=False)
    assert text == "hello world"


def test_A5_write_output(tmp_path):
    runner = STFTInferenceRunner()
    output = {"stft": {"type": "text", "value": "test"}}
    runner.write("utt1", output, str(tmp_path))  # str()に変換
    scp_path = Path(tmp_path) / "stft.scp"  # "spec" -> "stft"に修正
    assert scp_path.exists()
    content = scp_path.read_text().strip()
    assert content == "utt1 test"


def test_A6_manual_read_infer_write(test_audio_paths, tmp_path):
    path = str(test_audio_paths[0])
    runner = STFTInferenceRunner()
    model = runner.initialize_model()
    wav = runner.read("audio", path, stream=False)
    _, sample = runner.pre_inference(model, {"audio": wav})
    out = runner.inference_body(model, sample)
    runner.write("utt2", out, tmp_path)
    assert (tmp_path / "stft.scp").exists()


# === Streaming inference ===
def test_B1_streaming_on_example(test_audio_paths):
    path = test_audio_paths[0]
    runner = STFTInferenceRunner(stream=True)
    sample = {"audio_path": str(path)}
    out = runner.run_on_example("utt2", sample)
    assert "stft" in out and isinstance(out["stft"]["value"], str)
    assert "[]" not in out["stft"]["value"]


def test_B2_streaming_chunked_processing(test_audio_paths):
    path = test_audio_paths[0]
    x, sr = sf.read(path, dtype="float32")
    chunk_size = int(sr * 0.1)
    chunks = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]

    runner = STFTInferenceRunner(stream=True)
    model = runner.initialize_model()

    runner.pre_inference(model, {})
    outputs = [runner.inference_body(model, chunk) for chunk in chunks]
    result = runner.post_inference(model, outputs)

    assert "stft" in result
    assert isinstance(result["stft"]["value"], str)
    assert "[]" not in result["stft"]["value"]


@pytest.mark.parametrize(
    "chunk_sec, expected_min_chunks", [(0.01, 90), (0.1, 9), (0.5, 2)]
)
def test_B2_streaming_read_chunks(test_audio_paths, chunk_sec, expected_min_chunks):
    if len(test_audio_paths) == 0:
        pytest.skip("No test audio files available")

    path = str(test_audio_paths[0])
    runner = STFTInferenceRunner(stream=True)
    chunks = list(runner.read("audio", path, stream=True, chunk_sec=chunk_sec))

    assert isinstance(chunks, list)
    assert (
        len(chunks) >= expected_min_chunks
    ), f"Got only {len(chunks)} chunks for chunk_sec={chunk_sec}"
    assert all(isinstance(c, np.ndarray) for c in chunks)


def test_B3_read_text_streaming_chunks(tmp_path):
    file_path = tmp_path / "text.txt"
    file_path.write_text("abcdefghij")  # 長さ10

    runner = STFTInferenceRunner(stream=True)
    chunks = list(runner.read("text", str(file_path), stream=True, chunk_chars=3))

    assert chunks == ["abc", "def", "ghi", "j"]


def test_B4_post_inference_behavior():
    runner = STFTInferenceRunner(stream=True)
    model = runner.initialize_model()
    outputs = [
        {"stft": {"type": "text", "value": "chunk1"}},
        {"stft": {"type": "text", "value": "chunk2"}},
        {"stft": {"type": "text", "value": "chunk_final"}},
    ]
    result = runner.post_inference(model, outputs)
    assert result["stft"]["value"] == "chunk_final"


def test_B5_run_on_dataset_streaming(test_audio_paths, tmp_path):
    if len(test_audio_paths) == 0:
        pytest.skip("No test audio files available")

    path = str(test_audio_paths[0])

    # Hydra-compatible config with testset named 'testset'
    dataset_config = OmegaConf.create(
        {
            "_target_": "espnet3.data.DataOrganizer",
            "test": [
                {
                    "name": "testset",
                    "dataset": {
                        "_target_": "test.espnet3.test_data_organizer.DummyDataset",
                        "path": path,
                    },
                    "transform": {
                        "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                    },
                }
            ],
        }
    )

    runner.run()
    runner.compute_metrics(config.test)

    metrics_file = Path(config.decode_dir) / "test-a" / "metrics.json"
    assert metrics_file.exists()
    assert "dummy" in metrics_file.read_text()
