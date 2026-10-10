"""Keep pre-AQA imports and command lines usable during the task migration."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from espnet2.bin.aqa_inference import AqaInference
from espnet2.bin.aqa_train import main as train_main
from espnet2.bin.pack import get_parser
from espnet2.tasks.aqa import AqaTask


def test_task_and_cli_aliases():
    """Legacy names expose the same task, inference class, and train function."""
    name = "universa"
    task = importlib.import_module(f"espnet2.tasks.{name}")
    assert task.UniversaTask is AqaTask
    assert importlib.import_module(f"espnet2.bin.{name}_train").main is train_main
    module = importlib.import_module(f"espnet2.bin.{name}_inference")
    assert module.UniversaInference is AqaInference


@pytest.mark.parametrize(
    "module,symbol",
    [
        ("abs_universa", "AbsUniversa"),
        ("espnet_model", "ESPnetUniversaModel"),
        ("base", "UniversaBase"),
        ("base.loss", "masked_mse_loss"),
        ("base.loss", "masked_l1_loss"),
        ("base.universa_base", "UniversaBase"),
        ("ar_universa", "ARUniversa"),
        ("ar_universa.ar_universa", "ARUniversa"),
        ("ar_universa.data", "ARMetricCollateFn"),
        ("ar_universa.data", "ARMetricProcessor"),
        ("ar_universa.universa_beam_search", "ARUniVERSABeamSearch"),
        ("ar_universa.universa_beam_search", "MetricConstraintScorer"),
        ("metric_tokenizer.metric_tokenizer", "MetricTokenizer"),
        ("metric_tokenizer.metric_tokenizer", "AbsMetricTokenizer"),
    ],
)
def test_model_import_aliases(module, symbol):
    """Old model paths resolve to canonical classes rather than duplicate types."""
    old = importlib.import_module(f"espnet2.universa.{module}")
    new = importlib.import_module(f"espnet2.aqa.{module}")
    assert getattr(old, symbol) is getattr(new, symbol)


@pytest.mark.parametrize("name", ["aqa", "universa"])
def test_pack_aliases(name):
    """All task spellings retain identical model/config archive contents."""
    parser = get_parser()
    options = [
        "--outpath",
        "model.zip",
        "--model_file",
        "model.pth",
        "--train_config",
        "config.yaml",
    ]
    assert vars(parser.parse_args([name, *options])) == vars(
        parser.parse_args(["aqa", *options])
    )


@pytest.mark.parametrize("name", ["aqa", "universa"])
@pytest.mark.parametrize("command", ["train", "inference"])
@pytest.mark.execution_timeout(60)
def test_cli_help(name, command):
    """Canonical and legacy modules remain executable through python -m."""
    subprocess.run(
        [sys.executable, "-m", f"espnet2.bin.{name}_{command}", "--help"],
        check=True,
        capture_output=True,
        timeout=45,
    )


@pytest.mark.execution_timeout(60)
def test_evaluator_alias(tmp_path):
    """Old and new evaluator scripts produce identical strict JSON results."""
    scripts = Path(__file__).resolve().parents[3] / "egs2/TEMPLATE/asr1/pyscripts/utils"
    scores = tmp_path / "metric.scp"
    scores.write_text('a {"mos": 1.0}\nb {"mos": 2.0}\n')
    outputs = []
    for name in ("aqa", "universa"):
        output = tmp_path / f"{name}.json"
        subprocess.run(
            [
                sys.executable,
                str(scripts / f"{name}_eval.py"),
                "--ref_metrics",
                str(scores),
                "--pred_metrics",
                str(scores),
                "--out_file",
                str(output),
            ],
            check=True,
            timeout=45,
        )
        outputs.append(output.read_text())
    assert outputs[0] == outputs[1]
