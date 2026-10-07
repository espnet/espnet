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

    def run(name, *args, success=True, error_match=None):
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
        if error_match is not None:
            assert error_match in result.stderr, result.stdout + result.stderr
        return [json.loads(line) for line in calls.read_text().splitlines()]

    return root, run, stats, weights


def value(args, key):
    return args[args.index(key) + 1]


@pytest.mark.parametrize("name,stage", [("bagpiper", 5), ("bagpiper_tts", 3)])
@pytest.mark.parametrize("registered", [False, True])
def test_published_inference_command(recipe_tree, name, stage, registered):
    root, run, _, weights = recipe_tree
    config = root / "egs2" / name / "speechlm1/conf/train.yaml"
    option = (
        "--test-registered-specifier" if registered else "--test-unregistered-specifier"
    )
    specifier = (
        "dialogue:test" if registered else "dialogue:test:'/path with spaces/test.json'"
    )
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
        option,
        specifier,
    )
    assert len(calls) == 1
    assert calls[0][:2] == ["-m", "espnet2.speechlm.bin.inference"]
    assert value(calls[0], "--train-config") == str(config)
    assert value(calls[0], "--model-checkpoint") == str(weights)
    assert value(calls[0], option) == specifier
    other = (
        "--test-unregistered-specifier" if registered else "--test-registered-specifier"
    )
    assert other not in calls[0]


@pytest.mark.parametrize("name,stage", [("bagpiper", 5), ("bagpiper_tts", 3)])
@pytest.mark.parametrize("both_routes", [False, True])
def test_inference_requires_exactly_one_data_route(
    recipe_tree, name, stage, both_routes
):
    _, run, _, weights = recipe_tree
    data = (
        (
            "--test-registered-specifier",
            "dialogue:test",
            "--test-unregistered-specifier",
            "dialogue:test:/test.json",
        )
        if both_routes
        else ()
    )
    assert (
        run(
            name,
            "--stage",
            stage,
            "--export-path",
            weights,
            "--inference-config",
            "inference.yaml",
            *data,
            success=False,
            error_match=(
                "set only one" if both_routes else "set --test-unregistered-specifier"
            ),
        )
        == []
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


@pytest.mark.parametrize(
    "name,stage", [("bagpiper", 1), ("bagpiper", 3), ("bagpiper_tts", 1)]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_registered_training_and_resume(recipe_tree, name, stage, mixed):
    _, run, stats, weights = recipe_tree
    registered = {
        "train": "dialogue:train:0.5 text_only:text_train",
        "valid": "dialogue:valid",
    }
    unregistered = {
        "train": "dialogue:extra:'/path with spaces/train.json':0.5 text_only:extra_text:'/other path/text.json'",
        "valid": "dialogue:extra_valid:'/path with spaces/valid.json'",
    }
    data = []
    for split in ("train", "valid"):
        data.extend((f"--{split}-registered-specifier", registered[split]))
        if mixed:
            data.extend((f"--{split}-unregistered-specifier", unregistered[split]))
    args = ("--stage", stage, "--stop-stage", stage, "--stats-dir", stats, *data)
    for initialize in (True, False):
        initializer = ("--resume-path", weights) if initialize else ()
        calls = run(name, *args, *initializer)
        assert len(calls) == 1
        assert calls[0][:2] == ["-m", "torch.distributed.run"]
        if initialize:
            assert value(calls[0], "--resume-path") == str(weights)
        else:
            assert "--resume-path" not in calls[0]
        for split in ("train", "valid"):
            assert (
                value(calls[0], f"--{split}-registered-specifier") == registered[split]
            )
            if mixed:
                assert (
                    value(calls[0], f"--{split}-unregistered-specifier")
                    == unregistered[split]
                )
            else:
                assert f"--{split}-unregistered-specifier" not in calls[0]


@pytest.mark.parametrize(
    "name,stage", [("bagpiper", 1), ("bagpiper", 3), ("bagpiper_tts", 1)]
)
@pytest.mark.parametrize("missing", ["train", "valid"])
def test_registered_training_requires_both_splits(recipe_tree, name, stage, missing):
    _, run, stats, weights = recipe_tree
    provided = "valid" if missing == "train" else "train"
    error = (
        ("training specifier" if missing == "train" else "validation specifier")
        if name == "bagpiper"
        else f"set --{missing}-unregistered-specifier"
    )
    assert (
        run(
            name,
            "--stage",
            stage,
            "--stop-stage",
            stage,
            "--stats-dir",
            stats,
            "--resume-path",
            weights,
            f"--{provided}-registered-specifier",
            f"dialogue:{provided}",
            success=False,
            error_match=error,
        )
        == []
    )


@pytest.mark.parametrize("split", ["train", "valid"])
@pytest.mark.parametrize("route", ["registered", "unregistered"])
def test_standalone_sft_selects_each_split_without_mixing_scopes(
    recipe_tree, split, route
):
    _, run, stats, weights = recipe_tree
    other_route = "unregistered" if route == "registered" else "registered"
    specific = (
        "dialogue:sft"
        if route == "registered"
        else "dialogue:sft:'/SFT path/data.json'"
    )
    ordinary = (
        "dialogue:ordinary:/ordinary.json"
        if route == "registered"
        else "dialogue:ordinary"
    )
    calls = run(
        "bagpiper",
        "--stage",
        3,
        "--stop-stage",
        3,
        "--stats-dir",
        stats,
        "--resume-path",
        weights,
        f"--train-{other_route}-specifier",
        ordinary,
        f"--valid-{other_route}-specifier",
        ordinary,
        f"--sft-{split}-{route}-specifier",
        specific,
    )
    assert len(calls) == 1
    assert value(calls[0], f"--{split}-{route}-specifier") == specific
    assert f"--{split}-{other_route}-specifier" not in calls[0]
    fallback_split = "valid" if split == "train" else "train"
    assert value(calls[0], f"--{fallback_split}-{other_route}-specifier") == ordinary
    assert f"--{fallback_split}-{route}-specifier" not in calls[0]


def test_bagpiper_registered_curriculum_keeps_stage_data_separate(recipe_tree):
    _, run, stats, weights = recipe_tree
    calls = run(
        "bagpiper",
        "--stats-dir",
        stats,
        "--train-registered-specifier",
        "audio_to_text:pretrain:0.5",
        "--valid-registered-specifier",
        "audio_to_text:pretrain_valid",
        "--sft-stats-dir",
        stats,
        "--sft-train-registered-specifier",
        "dialogue:sft_train",
        "--sft-valid-registered-specifier",
        "dialogue:sft_valid",
        "--sft-train-unregistered-specifier",
        "dialogue:extra:'/SFT path/extra.json'",
        "--resume-path",
        weights,
        "--inference-config",
        "inference.yaml",
        "--test-registered-specifier",
        "dialogue:test",
    )
    assert len(calls) == 5
    for call in calls[:2]:
        assert (
            value(call, "--train-registered-specifier") == "audio_to_text:pretrain:0.5"
        )
        assert (
            value(call, "--valid-registered-specifier")
            == "audio_to_text:pretrain_valid"
        )
        assert "--train-unregistered-specifier" not in call
    assert value(calls[0], "--resume-path") == str(weights)
    assert value(calls[1], "--resume-path") == "exp/warmup/checkpoints/step_20"
    assert value(calls[2], "--resume-path") == "exp/pretrain/checkpoints/step_20"
    assert value(calls[2], "--train-registered-specifier") == "dialogue:sft_train"
    assert value(calls[2], "--valid-registered-specifier") == "dialogue:sft_valid"
    assert (
        value(calls[2], "--train-unregistered-specifier")
        == "dialogue:extra:'/SFT path/extra.json'"
    )
    assert "--valid-unregistered-specifier" not in calls[2]
    assert value(calls[4], "--test-registered-specifier") == "dialogue:test"
    assert "--test-unregistered-specifier" not in calls[4]


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
