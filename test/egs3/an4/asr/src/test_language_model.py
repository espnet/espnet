"""Exercise LM text, native stages and distributed restart handling locally."""

import gzip
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from omegaconf import OmegaConf

from egs3.an4.asr.src import language_model as module


@pytest.fixture
def lm_config(tmp_path):
    """Supply prepared ASR text, a tokenizer and optional external LM text."""
    (tmp_path / "train.txt").write_text("asr_1 HELLO\nasr_empty\n")
    (tmp_path / "valid.txt").write_text("valid_1 WORLD\n")
    (tmp_path / "native.yaml").write_text("lm: seq_rnn\n")
    (tmp_path / "tokens.txt").write_text("<blank>\n<unk>\nHELLO\n<sos/eos>\n")
    (tmp_path / "unigram.model").touch()
    archive = tmp_path / "external.gz"
    with gzip.open(archive, "wt") as stream:
        stream.write("EXTERNAL WORDS\n\nTHIRD LINE\n")
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "lm"),
            "train_text": str(tmp_path / "train.txt"),
            "valid_text": str(tmp_path / "valid.txt"),
            "test_text": str(tmp_path / "valid.txt"),
            "native_config": str(tmp_path / "native.yaml"),
            "tokenizer_dir": str(tmp_path),
            "ngpu": 1,
            "external_text": {"archive": str(archive), "url": "unused"},
        }
    )


def _rendezvous_path(arguments):
    uri = arguments[arguments.index("--dist_init_method") + 1]
    assert urlparse(uri).scheme == "file"
    return Path(unquote(urlparse(uri).path))


@pytest.fixture
def native_calls(monkeypatch):
    """Simulate stage outputs and FileStore creation without requiring GPUs."""
    calls = []

    def run(arguments, check):
        assert check is True
        calls.append(arguments)
        output = Path(arguments[arguments.index("--output_dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if "--collect_stats" in arguments:
            for split in ("train", "valid"):
                (output / split).mkdir(exist_ok=True)
                (output / split / "text_shape").write_text("u 2\n")
        elif "espnet2.bin.lm_train" in arguments:
            if "--dist_init_method" in arguments:
                rendezvous = _rendezvous_path(arguments)
                assert not rendezvous.exists()
                rendezvous.touch()
            (output / "valid.loss.ave.pth").touch()

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("ngpu", [1, 2])
def test_native_lm_pipeline_with_external_text(lm_config, native_calls, ngpu):
    """Keep source text IDs and vocabulary dimensions through all three stages."""
    lm_config.ngpu = ngpu
    checkpoint = module.train_language_model(lm_config)
    assert checkpoint.is_file()
    assert (checkpoint.parent / "lm_train.txt").read_text().splitlines() == [
        "asr_1 HELLO",
        "librispeech_lng_00000001 EXTERNAL WORDS",
        "librispeech_lng_00000003 THIRD LINE",
    ]
    assert (checkpoint.parent / "stats/train/text_shape.bpe").read_text() == "u 2,4\n"
    assert len(native_calls) == 3
    assert "espnet2.bin.lm_calc_perplexity" in native_calls[-1]
    training = native_calls[1]
    assert training[training.index("--ngpu") + 1] == str(ngpu)
    assert ("--dist_init_method" in training) == (ngpu > 1)
    if ngpu > 1:
        assert not _rendezvous_path(training).exists()


def test_distributed_resume_uses_fresh_rendezvous(lm_config, native_calls):
    """Resume model training without reusing a crashed launch's FileStore."""
    lm_config.ngpu = 2
    output = Path(lm_config.exp_dir)
    output.mkdir()
    stale = output / "distributed_init"
    stale.write_text("interrupted launch")
    for _ in range(2):
        module.train_language_model(lm_config)
    paths = []
    for training in (native_calls[1], native_calls[4]):
        assert training[training.index("--resume") + 1] == "true"
        assert training[training.index("--multiprocessing_distributed") + 1] == "true"
        path = _rendezvous_path(training)
        assert not path.exists()
        paths.append(path)
    assert paths[0] != paths[1]
    assert stale.read_text() == "interrupted launch"


@pytest.mark.parametrize("failed_stage", [0, 1, 2])
def test_native_failure_stops_pipeline(
    lm_config, native_calls, monkeypatch, failed_stage
):
    """Propagate stage failures and remove this launch's rendezvous on failure."""
    lm_config.ngpu = 2
    run = module.subprocess.run

    def fail(arguments, check):
        run(arguments, check)
        if len(native_calls) == failed_stage + 1:
            raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.train_language_model(lm_config)
    assert len(native_calls) == failed_stage + 1
    if failed_stage >= 1:
        assert not _rendezvous_path(native_calls[1]).exists()
