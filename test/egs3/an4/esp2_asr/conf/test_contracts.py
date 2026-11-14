"""Resolve all public roles and retain recipe-specific decoding/scoring choices."""

import logging
from pathlib import Path

from hydra.utils import get_class
from omegaconf import OmegaConf

from espnet3.contract.stages import roles, stage_names
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.esp2_lm.system import ESP2LMSystem
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.run_utils import apply_training_experiment_context

RECIPE = Path(__file__).resolve().parents[5] / "egs3/an4/esp2_asr"


def test_all_roles_resolve_and_keep_shared_inference(tmp_path):
    """Resolve publication/demo too, and pair metric refs with the infer dataset."""
    configs = {}
    for role in roles(ASRSystem):
        filename = "training_sinc_rnn.yaml" if role == "training" else f"{role}.yaml"
        path = RECIPE / "conf" / filename
        if not path.exists():
            path = tmp_path / "egs3/an4/esp2_asr/conf" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        configs[f"{role}_config"] = load_and_merge_config(
            path,
            f"{role}.yaml",
            default_package="egs3.TEMPLATE.esp2_asr",
            resolve=False,
        )
    apply_training_experiment_context(log=logging.getLogger(__name__), **configs)
    for config in configs.values():
        OmegaConf.resolve(config)
    inference = configs["inference_config"]
    metrics = configs["metrics_config"]
    assert metrics.inference_dir == inference.inference_dir
    assert "output_fn" not in inference
    assert inference.input_key == "speech"
    assert inference.contract.inputs.speech.sample_rate == 16000
    assert inference.contract.inputs.speech.channels == 1
    assert set(inference.contract.outputs) == {"text"}
    assert get_class(inference.runner._target_).__name__ == "ASRInferenceRunner"
    assert inference.model.beam_size == 10
    assert inference.model.ctc_weight == 0.3
    assert inference.model.lm_weight == 0.1
    assert len(metrics.metrics) == 3
    for metric in metrics.metrics:
        cls = get_class(metric.metric._target_)
        assert {field.name for field in cls.inputs} == set(metric.inputs)
        assert metric.inputs == {"ref": "dataset:text", "hyp": "text"}
    assert metrics.metrics[2].metric.bpemodel.endswith("data/unigram_30/unigram.model")
    publication = configs["publication_config"]
    assert publication.inference_dir == inference.inference_dir
    assert publication.pack_model.files.language_model == inference.model.lm_file
    assert Path(publication.pack_model.readme).is_file()
    assert Path(configs["demo_config"].ui.app_script).is_file()


def test_lm_has_its_own_training_template_and_cli():
    """Resolve LM construction without inheriting acoustic model/default keys."""
    assert default_conf_package(ESP2LMSystem) == "egs3.TEMPLATE.esp2_lm"
    assert roles(ESP2LMSystem) == ("training",)
    assert stage_names(ESP2LMSystem) == ["collect_stats", "train"]
    config = load_and_merge_config(
        RECIPE / "conf/training_lm.yaml",
        "training.yaml",
        default_package=default_conf_package(ESP2LMSystem),
    )
    assert get_class(config.task).__name__ == "LMTask"
    assert set(config.model) == {"token_list", "lm", "lm_conf"}
    assert "create_dataset" not in config
    assert "tokenizer" not in config
    args = build_parser(ESP2LMSystem).parse_args(
        ["--stages", "collect_stats", "train", "--training_config", "lm.yaml"]
    )
    assert args.stages == ["collect_stats", "train"]
    assert not hasattr(args, "inference_config")
