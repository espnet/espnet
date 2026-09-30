"""Exercise recipe stages using the real shared utilities."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "egs2/TEMPLATE/audio_metric1/audio_metric.sh"
# Recipe subprocesses import the toolkit afresh on shared CI workers.
pytestmark = pytest.mark.execution_timeout(60)


@pytest.fixture
def recipe(tmp_path):
    """Build a minimal recipe without requiring Kaldi or a corpus download."""
    (tmp_path / "egs2/test").mkdir(parents=True)
    (tmp_path / "egs2/TEMPLATE").symlink_to(ROOT / "egs2/TEMPLATE")
    tmp_path = tmp_path / "egs2/test/audio_metric1"
    subprocess.run([str(DRIVER.with_name("setup.sh")), str(tmp_path)], check=True)
    (tmp_path / "path.sh").unlink()
    (tmp_path / "path.sh").write_text("export LC_ALL=C\n")
    (tmp_path / "cmd.sh").write_text(
        "train_cmd=run.pl\ncuda_cmd=run.pl\ndecode_cmd=run.pl\n"
    )

    def run(stage, *options):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        result = subprocess.run(
            [
                "bash",
                str(tmp_path / "audio_metric.sh"),
                "--stage",
                str(stage),
                "--stop_stage",
                str(stage),
                "--python",
                sys.executable,
                "--train_set",
                "train",
                "--valid_set",
                "valid",
                *options,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return tmp_path, run


def test_character_tokens(recipe):
    """Character tokenization reads the configured reference text."""
    path, run = recipe
    (path / "reference.txt").write_text("a ab\nb bc\n")
    run(5, "--token_type", "char", "--bpe_train_text", "reference.txt")
    tokens = (path / "data/token_list/char/tokens.txt").read_text().splitlines()
    assert tokens == ["<blank>", "<unk>", "b", "a", "c", "<sos/eos>"]


@pytest.mark.parametrize("use_ref_wav", ["true", "false"])
def test_metric_ids_and_duration_filter(recipe, use_ref_wav):
    """Filter real references, retain missing references, and preserve metric IDs."""
    path, run = recipe
    for dset in ("train", "valid"):
        data = path / "dump/raw/org" / dset
        data.mkdir(parents=True)
        keys = ["long", "missing", "present", "short"]
        (data / "utt2spk").write_text("".join(f"{key} spk\n" for key in keys))
        (data / "spk2utt").write_text("spk " + " ".join(keys) + "\n")
        (data / "wav.scp").write_text("".join(f"{key} /{key}.wav\n" for key in keys))
        (data / "text").write_text("".join(f"{key} text\n" for key in keys))
        (data / "feats_type").write_text("raw\n")
        (data / "metric.scp").write_text(
            "".join(f'{key} {{"mos": 1.0}}\n' for key in keys)
        )
        (data / "utt2num_samples").write_text(
            "long 32000\nmissing 32000\npresent 32000\nshort 10\n"
        )
        (data / "ref_wav.scp").write_text(
            "long /long.wav\nmissing None\npresent /present.wav\nshort /short.wav\n"
        )
        (data / "utt2num_samples.ref").write_text(
            "long 999999\npresent 32000\nshort 32000\n"
        )
    run(3)
    run(4, "--use_ref_wav", use_ref_wav)
    result = path / "dump/raw/train"
    expected = (
        ["missing", "present"]
        if use_ref_wav == "true"
        else ["long", "missing", "present"]
    )
    for name in ("wav.scp", "metric.scp"):
        assert [
            line.split()[0] for line in (result / name).read_text().splitlines()
        ] == expected
    if use_ref_wav == "true":
        assert (
            result / "ref_wav.scp"
        ).read_text() == "missing None\npresent /present.wav\n"
    assert (result / "metric2id").read_text() == "mos\n"


@pytest.mark.parametrize(
    "audio,text", [(False, False), (True, False), (False, True), (True, True)]
)
def test_training_reference_shapes(recipe, audio, text):
    """Every optional reference shape gets its corresponding fold length."""
    path, run = recipe
    recorder = path / "record-python"
    recorder.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > train-args.txt\n')
    recorder.chmod(0o755)
    run(
        7,
        "--python",
        str(recorder),
        "--use_ref_wav",
        str(audio).lower(),
        "--use_ref_text",
        str(text).lower(),
    )
    args = (path / "train-args.txt").read_text().splitlines()
    shapes = [args[i + 1] for i, arg in enumerate(args) if arg == "--train_shape_file"]
    folds = [int(args[i + 1]) for i, arg in enumerate(args) if arg == "--fold_length"]
    assert len(shapes) == len(folds) == 1 + audio + text
    assert [Path(shape).name for shape in shapes] == ["audio_shape"] + (
        ["ref_audio_shape"] if audio else []
    ) + (["ref_text_shape"] if text else [])
    assert folds == [256000] + ([256000] if audio else []) + ([150] if text else [])
