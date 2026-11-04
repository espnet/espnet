"""Tests for egs3/TEMPLATE/esp2_s2t.

A recipe runs through ``launch(ESP2S2TSystem)``, which merges the recipe's
configs over this package's conf/. These pin that package and the defaults it
supplies.
"""

import pytest

# Imported here rather than inside the tests: the first of these pulls in torch
# and lightning, which takes longer than the suite's per-test timeout allows.
from espnet3.systems.esp2_s2t.system import ESP2S2TSystem  # noqa: E402
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_default_config

PACKAGE = "egs3.TEMPLATE.esp2_s2t"


def test_a_recipe_merges_over_this_template_not_the_asr_one():
    """launch() reads its defaults from the package named after the system.

    Inheriting the ASR template's defaults instead is silent: the configs still
    load, and the recipe picks up a tokenizer block that belongs to another
    system.
    """
    assert default_conf_package(ESP2S2TSystem) == PACKAGE


def test_train_tokenizer_is_not_a_stage_this_system_offers():
    """An S2T vocabulary ships with the checkpoint, so there is none to train.

    --stages accepts exactly the declared names, so a stage left out of the
    declaration cannot be reached at all.
    """
    assert "train_tokenizer" not in [s.name for s in ESP2S2TSystem.stages]
    with pytest.raises(SystemExit):
        build_parser(ESP2S2TSystem).parse_args(["--stages", "train_tokenizer"])


@pytest.mark.parametrize(
    "config_name",
    ["training.yaml", "inference.yaml", "metrics.yaml", "publication.yaml"],
)
def test_every_config_a_recipe_can_pass_has_a_default_here(config_name):
    """launch() merges each of these, so each needs a default to merge over.

    demo is the one role without one: this system ships no demo app.
    """
    assert load_default_config(config_name, PACKAGE) is not None


def test_the_training_default_names_the_s2t_task_and_no_tokenizer():
    """The two places this template differs from a generic one."""
    cfg = load_default_config("training.yaml", PACKAGE)
    assert cfg.task == "espnet3.systems.esp2_s2t.task.S2TTask"
    assert "tokenizer" not in cfg
    # The create_dataset stage prefers a recipe's DatasetBuilder, so the hook is
    # empty rather than pointing at a module a recipe does not ship. Empty and
    # not null: the stage reads it as dict(getattr(config, "create_dataset",
    # {})), and that default applies only when the key is absent, so a null
    # reaches dict() and raises before any builder method runs.
    assert dict(cfg.create_dataset) == {}


def test_the_training_default_carries_the_dataloader_plumbing():
    """A recipe's dataloader block is a partial overlay on this one.

    Recipes override the batch settings and nothing else, so the iterator
    target, the collate function and the sharding have to be here.
    """
    cfg = load_default_config("training.yaml", PACKAGE)
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert (
        cfg.dataloader.collate_fn._target_ == "espnet2.train.collate_fn.CommonCollateFn"
    )
    assert (
        cfg.dataloader.train.iter_factory._target_
        == "espnet2.iterators.sequence_iter_factory.SequenceIterFactory"
    )
    assert cfg.dataloader.train.total_shards == 1


def test_the_metric_defaults_are_the_utterance_group_pair():
    """The system's own metrics, and no symbol to score the wrong model with."""
    cfg = load_default_config("metrics.yaml", PACKAGE)
    targets = [entry.metric._target_ for entry in cfg.metrics]
    assert targets == [
        "espnet3.systems.esp2_s2t.metrics.cpwer.UtteranceGroupCpWER",
        "espnet3.systems.esp2_s2t.metrics.der.UtteranceGroupDER",
    ]
    for entry in cfg.metrics:
        assert "speaker_change_symbol" not in entry.metric
        assert dict(entry.inputs) == {"ref": "dataset:text", "hyp": "hyp"}


def test_the_inference_default_is_the_sot_runner_and_contract():
    """Both recipes inherit these, so they are stated once here."""
    cfg = load_default_config("inference.yaml", PACKAGE)
    assert cfg.model._target_ == (
        "espnet3.systems.esp2_s2t.src.s2t_inference.Speech2Text"
    )
    assert cfg.runner._target_ == (
        "espnet3.systems.esp2_s2t.src.inference_runner.ESP2S2TInferenceRunner"
    )
    assert cfg.input_key == "speech"
    assert cfg.batch_size is None
    assert list(cfg.contract.outputs) == ["hyp"]
    assert "output_fn" not in cfg
