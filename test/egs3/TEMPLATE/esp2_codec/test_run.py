"""Tests for the egs3/TEMPLATE/esp2_codec default configs and the recipe's launch."""

from pathlib import Path

from hydra.utils import get_class

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.stages import stage_names
from espnet3.systems.esp2_codec.system import ESP2CodecSystem
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

PACKAGE = "egs3.TEMPLATE.esp2_codec"
TEMPLATE_DIR = Path(__file__).resolve().parents[4] / "egs3" / "TEMPLATE" / "esp2_codec"


def test_the_system_runs_the_default_stages_from_this_template() -> None:
    """The codec system keeps the nine default stages and reads this package."""
    assert stage_names(ESP2CodecSystem) == [
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
    assert default_conf_package(ESP2CodecSystem) == PACKAGE


def test_parser_has_one_config_option_per_role() -> None:
    """The launcher takes a config path for each of the five roles."""
    parser = build_parser(ESP2CodecSystem)
    args = parser.parse_args(
        ["--stages", "infer", "--training_config", "conf/training_encodec.yaml"]
    )
    assert args.stages == ["infer"]
    assert args.training_config == Path("conf/training_encodec.yaml")
    for role in ("inference", "metrics", "publication", "demo"):
        assert getattr(args, f"{role}_config") is None


def test_training_scaffold_leaves_the_codec_to_the_recipe() -> None:
    """The template names the keys; the task, data and model come from a recipe."""
    config = load_default_config("training.yaml", PACKAGE)
    assert config.task is None
    assert config.dataset is None
    assert config.model is None
    assert config.trainer.accelerator == "auto"
    assert config.create_dataset == {}


def test_inference_defaults_wire_audiocoding_to_the_codec_runner() -> None:
    """The template owns the backend, the runner, input_key and the contract."""
    config = load_default_config("inference.yaml", PACKAGE)
    assert config.model._target_ == "espnet2.bin.gan_codec_inference.AudioCoding"
    assert config.input_key == "audio"
    assert config.batch_size is None
    assert config.dataset._recursive_ is False
    runner = get_class(config.runner._target_)
    assert runner.__name__ == "CodecInferenceRunner"

    contract = load_inference_contract(config.contract)
    contract.validate_input_keys(config.input_key)
    (audio,) = contract.inputs
    (wav,) = contract.outputs
    assert (audio.kind, audio.channels, audio.options["sample_rate"]) == (
        "audio",
        1,
        24000,
    )
    assert audio.options["normalize"] == ["mean_channels"]
    assert (wav.kind, wav.channels, wav.options["sample_rate"]) == ("audio", 1, 24000)
    assert config.output_artifacts.wav.sample_rate == wav.options["sample_rate"]


def test_recipe_overlay_keeps_the_template_runner_and_contract(tmp_path: Path) -> None:
    """A recipe names only its test set and checkpoint; the rest is inherited."""
    overlay = tmp_path / "inference.yaml"
    overlay.write_text(
        "dataset:\n"
        "  test:\n"
        "    - name: test\n"
        "      data_src_args: {split: test, inference: true}\n"
        "model:\n"
        "  train_config: exp/run/config.yaml\n"
        "  model_file: exp/run/last.ckpt\n",
        encoding="utf-8",
    )
    config = load_and_merge_config(overlay, "inference.yaml", default_package=PACKAGE)
    assert config.model._target_ == "espnet2.bin.gan_codec_inference.AudioCoding"
    assert config.model.model_file == "exp/run/last.ckpt"
    assert config.dataset.test[0].name == "test"
    assert "contract" in config and "output_fn" not in config
    assert config.runner._target_.endswith("CodecInferenceRunner")


def test_metrics_defaults_score_with_versa_against_the_test_set() -> None:
    """The template declares the VERSA metric; a recipe adds only its test sets."""
    config = load_default_config("metrics.yaml", PACKAGE)
    (entry,) = config.metrics
    assert get_class(entry.metric._target_).__name__ == "VersaMetric"
    assert [m.name for m in entry.metric.score_config] == [
        "signal_metric",
        "pseudo_mos",
    ]
    assert list(entry.metric.outputs) == [
        "sdr",
        "si_snr",
        "ci_sdr",
        "utmos",
        "dns_overall",
        "dns_p808",
        "plcmos",
    ]
    assert entry.inputs == {"wav": "wav", "ref": "dataset:wav_path"}


def test_demo_defaults_build_the_widgets_from_the_contract() -> None:
    """No ui.inputs/outputs are declared; the guide shown above them is packed."""
    config = load_default_config("demo.yaml", PACKAGE)
    assert config.model.trust_user_code is False
    assert "inputs" not in config.ui and "outputs" not in config.ui
    assert config.ui.app_script == "src/app.py"
    guide = TEMPLATE_DIR.parents[2] / config.ui.description
    assert guide.is_file()
    assert (TEMPLATE_DIR / "src" / "app.py").is_file()


def test_publication_defaults_pack_a_lean_bundle() -> None:
    """Checkpoints other than the registered file and all logs are excluded."""
    config = load_default_config("publication.yaml", PACKAGE)
    assert config.pack_model.out_dir == f"{config.exp_dir}/model_pack"
    for pattern in ("last.ckpt", "**/step*.ckpt", "**/epoch*.ckpt", "**/*.log"):
        assert pattern in list(config.pack_model.exclude)
