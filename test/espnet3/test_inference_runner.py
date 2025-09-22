from pathlib import Path
from typing import Union

import numpy as np
import pytest
import soundfile as sf
import torch

# from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from espnet3.inference.inference_runner import InferenceRunner

# ===============================================================
# Test Case Summary for STFTInferenceRunner
# ===============================================================
#
# Offline Processing Tests
# | Test Name                  | Description                                                           | # noqa: E501
# |---------------------------|-----------------------------------------------------------------------| # noqa: E501
# | test_A1_offline_on_example| Tests running inference on a full audio file in offline mode          | # noqa: E501
# | test_A3_read_audio_offline| Validates audio reading works for multiple files                      | # noqa: E501
# | test_A4_read_text_offline | Confirms reading plain text input offline                            | # noqa: E501
# | test_A5_write_output      | Ensures output is correctly written to stft.scp                      | # noqa: E501
# | test_A6_manual_read_infer_write | Manual read, inference, and write in offline mode             | # noqa: E501
# | test_A7_image_output_type       | Checks that image output is correctly written to PNG and scp         | # noqa: E501
# | test_A8_audio_output_type       | Checks that audio output is correctly written to FLAC and scp        | # noqa: E501
#
# Streaming Processing Tests
# | Test Name                        | Description                                                      | # noqa: E501
# |----------------------------------|------------------------------------------------------------------| # noqa: E501
# | test_B1_streaming_on_example     | Tests running full inference in streaming mode                  | # noqa: E501
# | test_B2_streaming_chunked_processing | Simulates chunk-by-chunk streaming inference                | # noqa: E501
# | test_B2_streaming_read_chunks    | Tests chunking of audio input based on duration                 | # noqa: E501
# | test_B3_read_text_streaming_chunks | Splits text input into chunks in streaming mode              | # noqa: E501
# | test_B4_post_inference_behavior  | Ensures last chunk's output is returned after streaming         | # noqa: E501
# | test_B5_run_on_dataset_streaming | Full dataset processing with config injection (Hydra-style)     | # noqa: E501


# === STFT Module ===
class StreamingSTFTModule:
    def __init__(self, stream: bool = False, n_fft=512, hop_length=128, win_length=512):
        self.stream = stream
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.n_freqs = n_fft // 2 + 1
        self.buffer = torch.zeros(n_fft - hop_length)

    def pre_inference(self, sample: dict):
        return sample

    def inference_body(self, chunk: Union[dict, np.ndarray]) -> dict:
        if isinstance(chunk, dict):
            chunk = chunk["audio"]
        chunk_tensor = torch.from_numpy(chunk).float()
        audio = torch.cat([self.buffer, chunk_tensor], dim=0)

        if audio.shape[0] < self.n_fft:
            self.buffer = audio
            return {"stft": {"type": "text", "value": "[]"}}

        num_frames = (audio.shape[0] - self.n_fft) // self.hop_length + 1
        if num_frames <= 0:
            self.buffer = audio
            return {"stft": {"type": "text", "value": "[]"}}

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        next_frame_start = self.hop_length * num_frames
        self.buffer = audio[next_frame_start:]

        # just return shape info for comparison
        return {
            "stft": {
                "type": "text",
                "value": str(list(stft.shape)),
            }
        }

    def post_inference(self, outputs: list) -> dict:
        return outputs[-1] if outputs else {"stft": {"type": "text", "value": "[]"}}


# === InferenceRunner Subclass ===
class STFTInferenceRunner(InferenceRunner):
    def __init__(self, stream: bool = False):
        super().__init__(stream=stream)

    def initialize_model(self, device=None):
        return StreamingSTFTModule(stream=self.stream)

    def pre_inference(self, model, sample):
        model.pre_inference(sample)
        return model, sample

    def inference_body(self, model, chunk):
        return model.inference_body(chunk)

    def post_inference(self, model, outputs):
        return model.post_inference(outputs)


# === Test Fixtures ===
@pytest.fixture(scope="module")
def test_audio_paths():
    base = Path("test_utils/espnet3/audio")
    assert base.exists(), f"Test audio directory not found: {base}"
    paths = sorted(base.glob("*.wav"))
    assert len(paths) > 0, f"No .wav files found in {base}"
    return paths


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


def test_A7_image_output_type(tmp_path):
    runner = STFTInferenceRunner()
    image_output = {"img": {"type": "image", "value": np.zeros((10, 10))}}
    runner.write("image_id1", image_output, tmp_path)
    assert (tmp_path / "img.scp").exists()
    assert (tmp_path / "data" / "img" / "image_id1.png").exists()


def test_A8_audio_output_type(tmp_path):
    runner = STFTInferenceRunner()
    audio_output = {"speech": {"type": "audio", "value": np.random.random(16000)}}
    runner.write("speech_id1", audio_output, tmp_path)
    assert (tmp_path / "speech.scp").exists()
    assert (tmp_path / "data" / "speech" / "speech_id1.flac").exists()


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

    # Define a runner with dataset_config injected
    class STFTInferenceRunnerWithConfig(InferenceRunner):
        def __init__(self, stream: bool = True):
            super().__init__(stream=stream, dataset_config=dataset_config)

        def initialize_model(self, device):
            return StreamingSTFTModule(stream=self.stream)

        def pre_inference(self, model, sample):
            model.pre_inference(sample)
            return model, sample

        def inference_body(self, model, chunk):
            return model.inference_body(chunk)

        def post_inference(self, model, outputs):
            return model.post_inference(outputs)

    runner = STFTInferenceRunnerWithConfig(stream=True)
    runner.run_on_dataset("testset", tmp_path)

    out_scp = tmp_path / "stft.scp"
    assert out_scp.exists()
    content = out_scp.read_text()
    assert "0" in content
