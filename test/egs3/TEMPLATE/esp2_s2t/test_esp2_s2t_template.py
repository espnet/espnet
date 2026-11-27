"""Tests for egs3/TEMPLATE/esp2_s2t.

A recipe runs through ``launch(ESP2S2TSystem)``, which merges the recipe's
configs over this package's conf/. These tests check the package name and the
default configs in it.
"""

import pytest

# These imports are at module level. The first one imports torch and lightning,
# which takes longer than the per-test timeout allows.
from espnet3.systems.esp2_s2t.system import ESP2S2TSystem  # noqa: E402
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_default_config

PACKAGE = "egs3.TEMPLATE.esp2_s2t"


def test_a_recipe_merges_over_this_template_not_the_asr_one():
    """launch() reads the defaults for ESP2S2TSystem from egs3.TEMPLATE.esp2_s2t.

    The package name comes from the system's module name. If launch() used the ASR
    template's defaults, no error would be raised. The configs would still load,
    and the recipe would get a tokenizer block from another system.
    """
    assert default_conf_package(ESP2S2TSystem) == PACKAGE


def test_train_tokenizer_is_not_a_stage_this_system_offers():
    """ESP2S2TSystem has no train_tokenizer stage.

    An S2T vocabulary comes with the checkpoint, so there is no tokenizer to train.
    --stages accepts only the declared stage names, so an undeclared stage cannot
    run.
    """
    assert "train_tokenizer" not in [s.name for s in ESP2S2TSystem.stages]
    with pytest.raises(SystemExit):
        build_parser(ESP2S2TSystem).parse_args(["--stages", "train_tokenizer"])


@pytest.mark.parametrize(
    "config_name",
    ["training.yaml", "inference.yaml", "metrics.yaml", "publication.yaml"],
)
def test_every_config_a_recipe_can_pass_has_a_default_here(config_name):
    """The template has a default for each config that a recipe can pass.

    launch() merges each of these configs over its default. demo is the only role
    without a default, because this system has no demo app.
    """
    assert load_default_config(config_name, PACKAGE) is not None


def test_the_training_default_names_the_s2t_task_and_no_tokenizer():
    """The training default sets the S2T task and has no tokenizer block.

    These are the two places where this template differs from a generic one.
    """
    cfg = load_default_config("training.yaml", PACKAGE)
    assert cfg.task == "espnet3.systems.esp2_s2t.task.S2TTask"
    assert "tokenizer" not in cfg
    # The create_dataset stage prefers a recipe's DatasetBuilder, so the hook is
    # empty. It does not point to a module that a recipe does not have. The stage
    # reads the hook as dict(getattr(config, "create_dataset", {})), and that
    # default is used only when the key is absent. So a null hook would be passed
    # to dict(), which raises before any builder method runs.
    assert dict(cfg.create_dataset) == {}


def test_the_training_default_carries_the_dataloader_plumbing():
    """The training default has the dataloader settings that recipes do not set.

    A recipe's dataloader block is a partial overlay on this one. Recipes override
    only the batch settings, so the iterator target, the collate function and the
    sharding must be set here.
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


def test_the_metric_defaults_are_the_utterance_group_metrics():
    """The metric defaults are this system's utterance-group metrics.

    No default sets speaker_change_symbol. A default symbol could be used to score
    a model that was trained with another symbol.
    """
    cfg = load_default_config("metrics.yaml", PACKAGE)
    targets = [entry.metric._target_ for entry in cfg.metrics]
    assert targets == [
        "espnet3.systems.esp2_s2t.metrics.cpwer.UtteranceGroupCpWER",
        "espnet3.systems.esp2_s2t.metrics.der.UtteranceGroupDER",
        "espnet3.systems.esp2_s2t.metrics.speaker_count."
        "UtteranceGroupSpeakerCountAccuracy",
        "espnet3.systems.esp2_s2t.metrics.wer.UtteranceGroupWER",
    ]
    for entry in cfg.metrics:
        assert "speaker_change_symbol" not in entry.metric
        assert dict(entry.inputs) == {"ref": "dataset:text", "hyp": "hyp"}


def test_the_inference_default_is_the_sot_runner_and_contract():
    """The inference default sets the SOT Speech2Text, the runner and the contract.

    Both recipes inherit these values, so the template defines them once.
    """
    cfg = load_default_config("inference.yaml", PACKAGE)
    assert cfg.model._target_ == (
        "espnet3.systems.esp2_s2t.src.sot.s2t_inference.Speech2Text"
    )
    assert cfg.runner._target_ == (
        "espnet3.systems.esp2_s2t.src.inference_runner.ESP2S2TInferenceRunner"
    )
    assert cfg.input_key == "speech"
    assert cfg.batch_size is None
    assert list(cfg.contract.outputs) == ["hyp"]
    assert "output_fn" not in cfg
