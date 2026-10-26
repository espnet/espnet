"""The kNN-VC TEMPLATE: launcher stages and the shared default configs."""

from espnet3.contract.stages import stage_names
from espnet3.systems.knnvc.system import KNNVCSystem
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_default_config

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_launcher_exposes_the_declared_stages   | --stages offers the system's |
# |                          | stages and one --<role>_config per config role. |
# | test_default_configs_resolve_from_the_template | launch() finds the        |
# |                          | TEMPLATE package from the system module.        |
# | test_load_default_config_train_contains_expected_defaults |              |
# | test_load_default_config_infer_declares_the_contract | model, input_key, |
# |                          | idx_key, contract and runner; no output_fn.     |
# | test_load_default_config_metrics_reads_the_transcript_from_the_data |    |


def test_launcher_exposes_the_declared_stages() -> None:
    parser = build_parser(KNNVCSystem)
    names = stage_names(KNNVCSystem)
    assert names.index("prepare_features") < names.index("train")
    assert "collect_stats" not in names and "train_tokenizer" not in names
    # VC ships no demo UI, so the demo stages stay out of the stage list.
    assert names[-2:] == ["pack_model", "upload_model"]

    args = parser.parse_args(["--stages", "prepare_features", "infer"])
    assert args.stages == ["prepare_features", "infer"]
    roles = {
        action.dest for action in parser._actions if action.dest.endswith("_config")
    }
    assert roles == {
        "training_config",
        "inference_config",
        "metrics_config",
        "publication_config",
    }


def test_default_configs_resolve_from_the_template() -> None:
    assert default_conf_package(KNNVCSystem) == "egs3.TEMPLATE.knnvc"
    for name in ("training", "inference", "metrics", "publication"):
        assert load_default_config(f"{name}.yaml", "egs3.TEMPLATE.knnvc") is not None


def test_load_default_config_train_contains_expected_defaults() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.knnvc")
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert cfg.prepare_features.prematch is True
    assert cfg.prepare_features.topk == 4
    assert cfg.dataloader.train.iter_factory is None
    assert cfg.best_model_criterion[0][0] == "valid/mel_loss"


def test_load_default_config_infer_declares_the_contract() -> None:
    cfg = load_default_config("inference.yaml", "egs3.TEMPLATE.knnvc")
    assert cfg.model._target_ == "espnet3.systems.knnvc.src.model.KNNVCModel"
    assert list(cfg.input_key) == ["speech", "reference_speech", "target_speaker"]
    assert cfg.idx_key == "pair_id"
    assert cfg.batch_size is None
    assert "output_fn" not in cfg and "output_artifacts" not in cfg
    assert list(cfg.contract.imports) == ["egs3.TEMPLATE.knnvc.src.contracts"]
    assert list(cfg.contract.inputs) == list(cfg.input_key)
    assert cfg.contract.inputs.speech.kind == "audio"
    assert cfg.contract.inputs.speech.sample_rate == 16000
    assert cfg.contract.inputs.speech.channels == 1
    assert cfg.contract.inputs.reference_speech.kind == "reference_audio"
    assert cfg.contract.inputs.reference_speech.channels == 1
    assert cfg.contract.inputs.target_speaker.kind == "text"
    assert list(cfg.contract.outputs) == ["wav"]
    assert cfg.contract.outputs.wav.kind == "audio"
    assert cfg.contract.outputs.wav.sample_rate == 16000
    assert (
        cfg.runner._target_
        == "espnet3.systems.knnvc.src.inference_runner.KNNVCInferenceRunner"
    )
    assert (
        cfg.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )


def test_load_default_config_metrics_reads_the_transcript_from_the_data() -> None:
    cfg = load_default_config("metrics.yaml", "egs3.TEMPLATE.knnvc")
    (entry,) = cfg.metrics
    assert entry.metric._target_.endswith("intelligibility.ASRIntelligibility")
    assert dict(entry.inputs) == {"wav": "wav", "ref": "dataset:text"}
