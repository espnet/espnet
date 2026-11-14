"""Tests for the egs3/TEMPLATE/lace default configs and the recipe's launch."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import get_class

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.stages import stage_names
from espnet3.systems.lace.system import LACESystem
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

PACKAGE = "egs3.TEMPLATE.lace"
REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_DIR = REPO_ROOT / "egs3" / "TEMPLATE" / "lace"
RECIPE_CONF = REPO_ROOT / "egs3" / "libritts" / "lace" / "conf"


# ---------------- default configs ----------------


def test_default_training_config_has_no_task_or_init():
    """LACE models are always Hydra targets; `init` would clobber loaded weights."""
    config = load_default_config("training.yaml", PACKAGE)
    assert "task" not in config
    assert "init" not in config
    assert config.exp_dir == "./exp/training"  # ${recipe_dir}/exp/${self_name:}
    assert config.trainer.devices == 1


def test_inference_defaults_wire_the_codec_to_the_lace_runner():
    """The template owns the backend, the runner, input_key and the contract."""
    config = load_default_config("inference.yaml", PACKAGE)
    assert config.model._target_ == "espnet3.systems.lace.src.inference.LACEAudioCoding"
    assert config.input_key == "audio"
    assert config.batch_size is None
    assert "output_fn" not in config and "output_artifacts" not in config
    assert config.dataset._recursive_ is False
    assert (
        config.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert get_class(config.runner._target_).__name__ == "LACEInferenceRunner"

    contract = load_inference_contract(config.contract)
    contract.validate_input_keys(config.input_key)
    (audio,) = contract.inputs
    (resyn_audio,) = contract.outputs
    assert (audio.name, audio.kind, audio.channels) == ("audio", "audio", 1)
    assert audio.options["normalize"] == ["mean_channels"]
    assert (resyn_audio.name, resyn_audio.kind, resyn_audio.channels) == (
        "resyn_audio",
        "audio",
        1,
    )
    # No rate is pinned: the codec's own `sample_rate` serves 16 and 24 kHz alike.
    assert "sample_rate" not in audio.options
    assert "sample_rate" not in resyn_audio.options


def test_metrics_scaffold_has_no_default_scorers():
    """A recipe supplies the whole VERSA list; the template keeps none."""
    config = load_default_config("metrics.yaml", PACKAGE)
    assert list(config.metrics) == []


def test_demo_defaults_build_the_widgets_from_the_contract():
    """No ui.inputs/outputs are declared; the guide shown above them is packed."""
    config = load_default_config("demo.yaml", PACKAGE)
    assert config.model.trust_user_code is False
    assert "inputs" not in config.ui and "outputs" not in config.ui
    assert config.ui.app_script == "src/app.py"
    assert (REPO_ROOT / config.ui.description).is_file()
    assert (TEMPLATE_DIR / "src" / "app.py").is_file()


def test_publication_defaults_pack_a_lean_bundle():
    """Checkpoints other than the registered file and all logs are excluded."""
    config = load_default_config("publication.yaml", PACKAGE)
    assert config.pack_model.out_dir == f"{config.exp_dir}/model_pack"
    for pattern in ("last.ckpt", "**/step*.ckpt", "**/epoch*.ckpt", "**/*.log"):
        assert pattern in list(config.pack_model.exclude)


# ---------------- recipe overlays ----------------


def test_libritts_training_config_resolves():
    """The recipe training config merges over the template and targets ``LACE``."""
    config = load_and_merge_config(
        RECIPE_CONF / "training.yaml", "training.yaml", default_package=PACKAGE
    )
    assert config.get("task") is None
    assert config.model._target_ == "espnet3.systems.lace.src.lace.LACE"
    assert (
        config.model.pretrained_model_tag
        == "NewGame/libritts_codec_train_encodec_libritts"
    )
    assert "pretrained_model_dir" not in config.model
    assert config.create_dataset.recipe_dir == "."
    assert config.data_dir == "./data"
    for split in ("train", "valid"):
        (entry,) = config.dataset[split]
        assert "data_src" not in entry  # the recipe's own dataset package


def test_libritts_inference_config_keeps_the_template_runner_and_contract():
    """The recipe names its test set and run; backend, contract and runner inherit."""
    config = load_and_merge_config(
        RECIPE_CONF / "inference.yaml", "inference.yaml", default_package=PACKAGE
    )
    assert config.model._target_ == "espnet3.systems.lace.src.inference.LACEAudioCoding"
    assert config.model.train_config.endswith("/lace_model.yaml")
    assert config.model.rate == 0.5
    assert config.input_key == "audio"
    assert "contract" in config and "output_fn" not in config
    assert config.runner._target_.endswith("LACEInferenceRunner")
    (entry,) = config.dataset.test
    assert entry.data_src_args.inference is True


def test_libritts_metrics_config_reads_the_reference_from_the_data():
    """The metrics config binds ``wav`` to the model output and ``ref`` to the data."""
    config = load_and_merge_config(
        RECIPE_CONF / "metrics.yaml", "metrics.yaml", default_package=PACKAGE
    )
    (metric,) = config.metrics
    assert metric.metric._target_ == "espnet3.systems.lace.metrics.versa.VersaMetric"
    assert dict(metric.inputs) == {"wav": "resyn_audio", "ref": "dataset:wav_path"}
    assert "utmos" in metric.metric.outputs
    assert "sir" not in metric.metric.outputs and "sar" not in metric.metric.outputs


def test_libritts_metrics_config_binds_the_declared_metric_inputs():
    """``VersaMetric``'s declared ``wav``/``ref`` inputs are the bound sources."""
    from hydra.utils import instantiate

    from espnet3.contract.metrics import check_metric_inputs

    config = load_and_merge_config(
        RECIPE_CONF / "metrics.yaml", "metrics.yaml", default_package=PACKAGE
    )
    (entry,) = config.metrics
    metric = instantiate(entry.metric)
    # `wav` is the contract's audio output the infer stage writes as
    # resyn_audio.scp; `ref` is the test set's own waveform column.
    assert [f.name for f in metric.inputs] == ["wav", "ref"]
    assert check_metric_inputs(metric, entry.inputs) == {
        "wav": "resyn_audio",
        "ref": "dataset:wav_path",
    }
    assert [f.name for f in metric.outputs] == list(entry.metric.outputs)


def test_libritts_publication_config_ships_no_recipe_code():
    """The bundle registers the checkpoint and packs neither `src` nor `dataset`."""
    config = load_and_merge_config(
        RECIPE_CONF / "publication.yaml",
        "publication.yaml",
        default_package=PACKAGE,
    )
    assert config.pack_model.files.model_file.endswith("/last.ckpt")
    assert list(config.pack_model.include) == []


# ---------------- launch() ----------------


def test_launch_parser_exposes_the_lace_roles_and_stages():
    """``launch`` sees the LACE stages and the template conf package."""
    parser = build_parser(LACESystem)
    args = parser.parse_args(["--stages", "train", "--training_config", "x.yaml"])
    assert args.stages == ["train"]
    assert str(args.training_config) == "x.yaml"
    assert default_conf_package(LACESystem) == PACKAGE
    assert stage_names(LACESystem) == [
        "create_dataset",
        "collect_stats",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
        "pack_demo",
        "upload_demo",
    ]
