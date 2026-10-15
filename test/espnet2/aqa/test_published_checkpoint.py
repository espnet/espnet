"""Exercise the opt-in validator with small local checkpoint packages."""

import json
from test.espnet2.aqa.check_published_checkpoint import main
from test.espnet2.aqa.test_aqa_task import task_args
from test.espnet2.aqa.test_metric_tokenizer import token_info

import numpy as np
import pytest
import torch
import yaml

from espnet2.aqa.espnet_model import ESPnetUniversaModel
from espnet2.tasks.aqa import AqaTask


@pytest.fixture
def package(tmp_path):
    """Save model weights and portable metadata as a downloaded package would."""

    def create(arecho=False, references=False):
        """Write one tiny model package and return its validator arguments."""
        args = task_args(tmp_path)
        args.use_ref_audio = args.use_ref_text = references
        if references:
            args.token_list = ["<blank>", "<unk>", "<sos/eos>"]
            args.universa_conf["text_encoder_params"] = dict(
                num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
            )
        if arecho:
            args.universa = "ar_universa"
            args.metric2id = ["mos", "language"]
            args.metric2type = {"mos": "numerical", "language": "categorical"}
            args.metric_token_info = token_info()
            args.universa_conf["metric_decoder_params"] = dict(
                num_blocks=1, attention_heads=2, linear_units=16
            )
        model = AqaTask.build_model(args)
        torch.save(model.state_dict(), tmp_path / "model.pth")
        # Released packages can retain file-backed metadata relative to their root.
        (tmp_path / "metric2id").write_text("\n".join(args.metric2id) + "\n")
        args.metric2id = "metric2id"
        if references:
            (tmp_path / "tokens.txt").write_text("\n".join(args.token_list) + "\n")
            args.token_list = "tokens.txt"
        if arecho:
            (tmp_path / "metric2type").write_text(
                "mos numerical\nlanguage categorical\n"
            )
            args.metric2type = "metric2type"
            (tmp_path / "token_info.json").write_text(
                json.dumps(args.metric_token_info)
            )
            args.metric_token_info = "token_info.json"
        args.freeze_param = ["universa.audio_encoder"]
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(vars(args)))
        metrics = {"mos": 1.5, "language": "eng"} if arecho else {"mos": 1.5}
        return [
            str(tmp_path),
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pth",
            "--metrics",
            json.dumps(metrics),
        ]

    return create


@pytest.mark.parametrize("arecho", [False, True])
@pytest.mark.parametrize("references", [False, True])
def test_strict_load_finetune_and_inference(package, capsys, arecho, references):
    """Both architectures must update parameters and return valid predictions."""
    main(package(arecho, references))
    output = capsys.readouterr().out
    assert "STRICT_LOAD <All keys matched successfully>" in output
    assert "FINETUNE_STEP" in output and "PREDICTED_METRICS" in output


@pytest.mark.parametrize("metrics", ["{}", "[]", '{"mos": NaN}', '{"mos": -100}'])
def test_reject_invalid_labels(package, metrics):
    """Empty, nonfinite, and masked labels cannot validate a training step."""
    options = package()
    options[-1] = metrics
    with pytest.raises((SystemExit, ValueError)):
        main(options)


@pytest.mark.parametrize("prediction", [{}, {"mos": np.array([np.nan])}])
def test_reject_missing_or_nonfinite_predictions(package, monkeypatch, prediction):
    """A successful optimization step alone must not pass validation."""
    monkeypatch.setattr(ESPnetUniversaModel, "inference", lambda *a, **kw: prediction)
    with pytest.raises(ValueError, match="prediction"):
        main(package())


def test_reject_incompatible_checkpoint(package, tmp_path):
    """Missing weights must fail the strict load rather than silently initialize."""
    options = package()
    torch.save({}, tmp_path / "model.pth")
    with pytest.raises(RuntimeError, match="Missing key"):
        main(options)


@pytest.mark.parametrize("recipe", ["TEMPLATE", "mini_an4", "urgent24"])
def test_legacy_recipe_aliases(recipe):
    """Legacy directories and scripts resolve to the canonical AQA driver."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    alias = root / "egs2" / recipe / "uni_versa1"
    canonical = alias.with_name("aqa1")
    assert alias.is_symlink() and alias.resolve(strict=True) == canonical
    driver = alias / "uni_versa.sh"
    assert driver.is_symlink()
    assert driver.resolve(strict=True) == (canonical / "aqa.sh").resolve(strict=True)
