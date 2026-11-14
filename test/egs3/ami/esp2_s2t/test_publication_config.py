"""Tests for egs3/ami/esp2_s2t/conf/publication.yaml."""

import ami_sot_paths


def _config():
    """Load the config the way run.py does."""
    from espnet3.utils.config_utils import load_and_merge_config

    return load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / "publication.yaml",
        config_name="publication.yaml",
        default_package=ami_sot_paths.TEMPLATE_PACKAGE,
        resolve=True,
    )


def test_the_packed_bundle_carries_the_dataset_and_no_recipe_code():
    """The bundle needs the dataset module, and no recipe-local code.

    dataset/ supplies the builder the publication smoke check runs. The model,
    runner and preprocessor are installed with espnet under
    espnet3.systems.esp2_s2t, so a recipe src/ would be dead weight.
    """
    include = [str(p) for p in _config()["pack_model"]["include"]]
    assert any(
        p.rstrip("/").endswith("/dataset") or p == "dataset" for p in include
    ), include
    assert not any(p.rstrip("/").endswith("src") for p in include), include


def test_the_checkpoint_config_is_not_a_yaml_files_entry():
    """pack_model would write ${recipe_dir} into it, which espnet2 cannot read.

    espnet2 loads a checkpoint's config.yaml with yaml.safe_load, so an
    OmegaConf reference there stays a literal path. ESP2S2TSystem.pack_model
    writes the token list into the bundled copy instead.
    """
    assert not _config()["pack_model"].get("yaml_files")


def test_the_bundle_decodes_the_checkpoint_it_carries(monkeypatch):
    """By default infer reads the train stage's own output, which is bundled.

    A default pointing anywhere else would leave a packed model unable to find
    its weights once copied to another machine.
    """
    from espnet3.utils.config_utils import load_and_merge_config

    monkeypatch.delenv("AMI_SOT_TRAIN_CONFIG", raising=False)
    monkeypatch.delenv("AMI_SOT_MODEL_FILE", raising=False)
    config = load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / "inference.yaml",
        config_name="inference.yaml",
        default_package=ami_sot_paths.TEMPLATE_PACKAGE,
        resolve=False,
    )
    config.exp_dir = "exp/run"
    assert config.model.s2t_train_config == "exp/run/config.yaml"
    assert config.model.s2t_model_file == "exp/run/valid.loss.ave_1best.pth"


def test_checkpoints_are_kept_but_lightning_snapshots_are_not():
    exclude = [str(p) for p in _config()["pack_model"]["exclude"]]
    assert any("ckpt" in p for p in exclude), exclude
    assert not any(p.endswith(".pth") for p in exclude), exclude
