"""Exercise recipe and train.sh argument handling without launching GPU jobs."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="SpeechLM shell recipes require a POSIX shell",
)


@pytest.fixture
def recipe_tree(tmp_path):
    """Keep real launchers; record only the final Python process boundary."""
    source = Path(__file__).resolve().parents[3]
    root = tmp_path / "recipe tree"
    for name in ("bagpiper", "bagpiper_tts"):
        relative = Path("egs2") / name / "speechlm1"
        shutil.copytree(source / relative / "conf", root / relative / "conf")
        shutil.copy2(source / relative / "run.sh", root / relative / "run.sh")
    for relative in (
        "egs2/TEMPLATE/speechlm1/train.sh",
        "egs2/TEMPLATE/speechlm1/stage_utils.sh",
        "egs2/TEMPLATE/asr1/utils/parse_options.sh",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    recorder = tmp_path / "record_python"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, shutil, sys\n"
        "args = sys.argv[1:]\n"
        "def value(key): return args[args.index(key) + 1]\n"
        "with open(os.environ['RECIPE_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[:2] == ['-m', 'torch.distributed.run']:\n"
        "    output = pathlib.Path(value('--output-dir'))\n"
        "    checkpoint = output / 'checkpoints' / 'step_20'\n"
        "    checkpoint.mkdir(parents=True, exist_ok=True)\n"
        "    (checkpoint / '.metadata').touch()\n"
        "    shutil.copyfile(value('--train-config'), output / 'train.yaml')\n"
        "elif args[:2] == ['-m', 'espnet2.speechlm.bin.export_checkpoint']:\n"
        "    output = pathlib.Path(value('--output'))\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with output.open('x') as stream: stream.write('exported weights')\n"
    )
    recorder.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    stats = tmp_path / "length stats"
    stats.mkdir()
    weights = tmp_path / "public model.pt"
    weights.write_text("public weights")

    def run(name, *args, success=True):
        calls.write_text("")
        result = subprocess.run(
            [
                "bash",
                str(root / "egs2" / name / "speechlm1/run.sh"),
                "--python",
                str(recorder),
                *map(str, args),
            ],
            cwd=tmp_path,
            env={**os.environ, "RECIPE_CALLS": str(calls)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return [json.loads(line) for line in calls.read_text().splitlines()]

    return root, run, stats, weights


def value(args, key):
    return args[args.index(key) + 1]


@pytest.mark.parametrize("name,stage", [("bagpiper", 5), ("bagpiper_tts", 3)])
def test_published_inference_command(recipe_tree, name, stage):
    root, run, _, weights = recipe_tree
    config = root / "egs2" / name / "speechlm1/conf/train.yaml"
    calls = run(
        name,
        "--stage",
        stage,
        "--train-config",
        config,
        "--export-path",
        weights,
        "--inference-config",
        "inference.yaml",
        "--test-unregistered-specifier",
        "dialogue:test:/path with spaces/test.json",
    )
    assert len(calls) == 1
    assert calls[0][:2] == ["-m", "espnet2.speechlm.bin.inference"]
    assert value(calls[0], "--train-config") == str(config)
    assert value(calls[0], "--model-checkpoint") == str(weights)
    assert value(calls[0], "--test-unregistered-specifier").endswith(
        "/path with spaces/test.json"
    )


@pytest.mark.parametrize("name,stage", [("bagpiper", 3), ("bagpiper_tts", 1)])
def test_initialize_then_resume_training(recipe_tree, name, stage):
    root, run, stats, weights = recipe_tree
    config = root / "egs2" / name / "speechlm1/conf/train.yaml"
    args = (
        "--stage",
        stage,
        "--stop-stage",
        stage,
        "--stats-dir",
        stats,
        "--train-unregistered-specifier",
        "dialogue:train:/train.json",
        "--valid-unregistered-specifier",
        "dialogue:valid:/valid.json",
        "--train-config",
        config,
        "--output-dir",
        "exp/custom sft",
        "--train-args",
        "--wandb-mode offline --save-loader-state false",
    )
    calls = run(name, *args, "--resume-path", weights)
    assert len(calls) == 1
    assert calls[0][:2] == ["-m", "torch.distributed.run"]
    assert value(calls[0], "--resume-path") == str(weights)
    assert value(calls[0], "--output-dir") == "exp/custom sft"
    assert value(calls[0], "--stats-dir") == str(stats)
    assert value(calls[0], "--train-config") == str(config)
    assert value(calls[0], "--wandb-mode") == "offline"
    assert "--save-loader-state" not in calls[0]
    calls = run(name, *args)
    assert "--resume-path" not in calls[0]
    # An explicit initialization must also win when the output has a checkpoint.
    calls = run(name, *args, "--resume-path", weights)
    assert value(calls[0], "--resume-path") == str(weights)


def test_bagpiper_curriculum_and_resume(recipe_tree):
    _, run, stats, weights = recipe_tree
    args = (
        "--stats-dir",
        stats,
        "--train-unregistered-specifier",
        "audio_to_text:train:/pretrain.json",
        "--valid-unregistered-specifier",
        "audio_to_text:valid:/valid.json",
        "--sft-stats-dir",
        stats,
        "--sft-train-specifier",
        "dialogue:train:/sft.json",
        "--sft-valid-specifier",
        "dialogue:valid:/sft-valid.json",
        "--inference-config",
        "inference.yaml",
        "--test-unregistered-specifier",
        "dialogue:test:/test.json",
    )
    calls = run("bagpiper", *args, "--resume-path", weights)
    assert len(calls) == 5
    assert value(calls[0], "--resume-path") == str(weights)
    assert value(calls[1], "--resume-path") == "exp/warmup/checkpoints/step_20"
    assert value(calls[2], "--resume-path") == "exp/pretrain/checkpoints/step_20"
    assert value(calls[2], "--train-unregistered-specifier") == (
        "dialogue:train:/sft.json"
    )
    assert value(calls[3], "--checkpoint-dir") == "exp/sft/checkpoints/step_20"
    assert calls[4][:2] == ["-m", "espnet2.speechlm.bin.inference"]
    calls = run("bagpiper", *args, "--stop-stage", 3)
    assert len(calls) == 3
    assert all("--resume-path" not in call for call in calls)


@pytest.mark.parametrize("name,stage", [("bagpiper", 4), ("bagpiper_tts", 2)])
def test_export_selection_and_existing_file(recipe_tree, name, stage):
    root, run, _, _ = recipe_tree
    recipe = root / "egs2" / name / "speechlm1"
    for step in ("9", "20", "30", "invalid"):
        path = recipe / "exp/sft/checkpoints" / f"step_{step}"
        path.mkdir(parents=True)
        if step != "30":
            (path / ".metadata").touch()
    calls = run(name, "--stage", stage, "--stop-stage", stage)
    assert value(calls[0], "--checkpoint-dir") == "exp/sft/checkpoints/step_20"
    # Existing weights must not silently stand in for a newer checkpoint.
    exported = recipe / "exp/sft/export/model.pt"
    original = exported.read_bytes()
    assert run(name, "--stage", stage, "--stop-stage", stage, success=False) == []
    assert exported.read_bytes() == original
    calls = run(
        name,
        "--stage",
        stage,
        "--stop-stage",
        stage,
        "--checkpoint-dir",
        "exp/sft/checkpoints/step_9",
        "--export-path",
        "exp/older/model.pt",
    )
    assert value(calls[0], "--checkpoint-dir") == "exp/sft/checkpoints/step_9"


@pytest.mark.parametrize("name", ["bagpiper", "bagpiper_tts"])
@pytest.mark.parametrize(
    "args",
    [
        ("--stage", "invalid"),
        ("--stage", "0"),
        ("--stage", "3", "--stop-stage", "1"),
        ("--stop-stage", "6"),
        ("unexpected",),
    ],
)
def test_invalid_stage_or_arguments(recipe_tree, name, args):
    _, run, _, _ = recipe_tree
    assert run(name, *args, success=False) == []


def test_missing_predecessor_and_secondary_node(recipe_tree):
    _, run, stats, _ = recipe_tree
    args = (
        "--stats-dir",
        stats,
        "--train-unregistered-specifier",
        "dialogue:train:/train.json",
        "--valid-unregistered-specifier",
        "dialogue:valid:/valid.json",
    )
    assert run("bagpiper", "--stage", 2, "--stop-stage", 2, *args, success=False) == []
    calls = run(
        "bagpiper",
        *args,
        "--sft-stats-dir",
        stats,
        "--sft-train-specifier",
        "dialogue:train:/sft.json",
        "--sft-valid-specifier",
        "dialogue:valid:/sft-valid.json",
        "--num-nodes",
        2,
        "--node-rank",
        1,
        "--master-addr",
        "localhost",
    )
    assert len(calls) == 3
    assert all(call[:2] == ["-m", "torch.distributed.run"] for call in calls)
