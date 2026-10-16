"""Tests for LID stage configuration, paths, and publication artifacts."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.systems.esp2_lid.system import LIDSystem
from espnet3.utils.config_utils import load_and_merge_config, load_default_config
from espnet3.utils.publication_utils import pack_model


def test_lid_template_configs_define_inference_measure_and_publication():
    """Keep the stage defaults and optional global seed usable."""
    training = load_default_config("training.yaml", "egs3.TEMPLATE.esp2_lid")
    inference = load_default_config("inference.yaml", "egs3.TEMPLATE.esp2_lid")
    metrics = load_default_config("metrics.yaml", "egs3.TEMPLATE.esp2_lid")
    publication = load_default_config("publication.yaml", "egs3.TEMPLATE.esp2_lid")

    assert inference.provider._target_.endswith("InferenceProvider")
    assert inference.runner._target_.endswith("InferenceRunner")
    assert training.seed is None
    assert training.dataloader.train.iter_factory.seed == 0
    assert training.dataloader.valid.iter_factory.seed == 0
    assert training.trainer.accumulate_grad_batches == 1
    assert inference.model._target_ == "espnet3.systems.esp2_lid.inference.Inference"
    assert metrics.metrics[0].metric._target_.endswith(
        "esp2_lid.metrics.accuracy.Accuracy"
    )
    for metric in metrics.metrics:
        assert dict(metric.inputs) == {"ref": "dataset:lid_labels", "hyp": "language"}
    assert publication.pack_model.readme.endswith("src/hf_model_readme.md")
    assert [stage.name for stage in LIDSystem.stages] == [
        "create_dataset",
        "collect_stats",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
    ]


@pytest.mark.parametrize("name", ["inference.yaml", "inference_prepared.yaml"])
def test_voxlingua_inference_configs_use_the_lid_inference(name):
    """Name the system's Inference, so infer writes the declared language.scp."""
    config = load_and_merge_config(
        Path("egs3/voxlingua107/esp2_lid/conf") / name,
        config_name="inference.yaml",
        default_package="egs3.TEMPLATE.esp2_lid",
        resolve=False,
    )

    assert config.model._target_ == "espnet3.systems.esp2_lid.inference.Inference"
    for key in ("input_key", "output_fn", "output_keys"):
        assert key not in config


def test_voxlingua_metrics_read_the_reference_from_the_dataset():
    config = load_and_merge_config(
        Path("egs3/voxlingua107/esp2_lid/conf/metrics.yaml"),
        config_name="metrics.yaml",
        default_package="egs3.TEMPLATE.esp2_lid",
        resolve=False,
    )

    for metric in config.metrics:
        assert dict(metric.inputs) == {"ref": "dataset:lid_labels", "hyp": "language"}


def test_voxlingua_training_overrides_default_sampler():
    """Connect the sampler and preprocessor to combined statistics and recipe seed."""
    config = load_and_merge_config(
        Path("egs3/voxlingua107/esp2_lid/conf/training.yaml"),
        config_name="training.yaml",
        resolve=True,
    )

    assert config.dataloader.train.iter_factory.batches.type == "catpow"
    assert config.dataloader.valid.iter_factory.batches.type == "catpow"
    for mode in ("train", "valid"):
        factory = config.dataloader[mode].iter_factory
        assert factory.seed == config.seed
        assert (
            factory.batches.category2utt_file
            == f"{config.stats_dir}/{mode}/category2utt"
        )
    assert config.dataset.preprocessor.lang2utt == f"{config.stats_dir}/train/lang2utt"
    for mode in ("train", "valid"):
        assert config.dataset[mode][0].data_src_args.data_dir == (
            f"{config.data_dir}/voxlingua107"
        )
    assert config.trainer.accumulate_grad_batches == 2


def test_voxlingua_model_pack_includes_lang2utt(tmp_path):
    """Bundle the collected training inventory and rewrite its references."""
    recipe_dir = tmp_path / "recipe"
    exp_dir = recipe_dir / "exp" / "run"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text("dummy: true\n", encoding="utf-8")

    lang2utt = recipe_dir / "exp" / "stats" / "train" / "lang2utt"
    lang2utt.parent.mkdir(parents=True)
    lang2utt.write_text("eng 0\nfra 1\n", encoding="utf-8")

    publication = load_and_merge_config(
        Path("egs3/voxlingua107/esp2_lid/conf/publication.yaml"),
        config_name="publication.yaml",
        default_package="egs3.TEMPLATE.esp2_lid",
        resolve=False,
    )
    publication.recipe_dir = str(recipe_dir)
    publication.pack_model.out_dir = str(tmp_path / "pack")
    publication.pack_model.readme = None
    OmegaConf.resolve(publication)

    training = OmegaConf.create(
        {
            "recipe_dir": str(recipe_dir),
            "exp_dir": str(exp_dir),
            "dataset": {
                "preprocessor": {"lang2utt": str(lang2utt)},
            },
        }
    )
    inference = OmegaConf.create(
        {
            "recipe_dir": str(recipe_dir),
            "model": {"lang2utt": str(lang2utt)},
        }
    )

    out_dir = pack_model(
        training_config=training,
        publication_config=publication,
        inference_config=inference,
    )

    assert (out_dir / "exp/stats/train/lang2utt").read_text(encoding="utf-8") == (
        "eng 0\nfra 1\n"
    )
    bundle_inference_path = out_dir / "conf" / "inference.yaml"
    bundle_training_path = out_dir / "conf" / "training.yaml"
    assert "lang2utt: ${recipe_dir}/exp/stats/train/lang2utt" in (
        bundle_inference_path.read_text(encoding="utf-8")
    )
    assert "lang2utt: ${recipe_dir}/exp/stats/train/lang2utt" in (
        bundle_training_path.read_text(encoding="utf-8")
    )
    assert OmegaConf.load(bundle_inference_path).model.lang2utt == (
        "./exp/stats/train/lang2utt"
    )
