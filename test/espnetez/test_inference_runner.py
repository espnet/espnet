# tests/test_inference_runner.py
import shutil
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
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


@pytest.fixture
def clean_decode_dir(config):
    decode_dir = Path(config.decode_dir)
    if decode_dir.exists():
        shutil.rmtree(decode_dir)
    yield decode_dir
    if decode_dir.exists():
        shutil.rmtree(decode_dir)


def test_inference_runner_from_yaml(config, clean_decode_dir):
    runner = InferenceRunner(
        config=config,
        model_config=config.model,
        decode_dir=Path(config.decode_dir),
    )

    runner.run()
    runner.compute_metrics(config.test)

    metrics_file = Path(config.decode_dir) / "test-a" / "metrics.json"
    assert metrics_file.exists()
    assert "dummy" in metrics_file.read_text()
