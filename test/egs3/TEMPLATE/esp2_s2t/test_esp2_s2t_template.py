"""Tests for egs3/TEMPLATE/esp2_s2t.

The template reuses the generic argument parsing and stage loop from the ASR
template. What it must not reuse is the ASR template's default configs, so
these pin which package a recipe's configs are merged over.
"""

import argparse

import pytest

from espnet3.utils.config_utils import load_default_config

PACKAGE = "egs3.TEMPLATE.esp2_s2t"


def _args(**overrides):
    """Build the Namespace main() reads, with every config absent."""
    fields = dict(
        stages=["train"],
        training_config=None,
        inference_config=None,
        metrics_config=None,
        publication_config=None,
        demo_config=None,
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _record_packages(monkeypatch, module):
    """Record the default_package of every config load ``module.main`` makes.

    main() cannot get past loading without real configs, and where it loaded
    them from is the whole subject here, so the failure after that is expected.
    """
    seen = []

    def _load(config_path, config_name, default_package=None, resolve=True):
        seen.append(default_package)
        return None

    monkeypatch.setattr(module, "load_and_merge_config", _load)
    return seen


def test_a_recipe_merges_over_this_template_not_the_asr_one(monkeypatch):
    """An esp2_s2t recipe's configs must be merged over this template's conf/.

    Inheriting the ASR template's defaults instead is silent: the configs still
    load, and the recipe picks up a tokenizer block and a create_dataset hook
    that belong to another system.
    """
    from egs3.TEMPLATE.esp2_s2t import run as s2t

    seen = _record_packages(monkeypatch, s2t)
    try:
        s2t.main(_args(), system_cls=object, stages=s2t.DEFAULT_STAGES)
    except Exception:
        pass

    assert seen, "main() loaded no configs"
    assert set(seen) == {PACKAGE}


def test_this_templates_runner_is_its_own_not_the_asr_one():
    """The runner is a copy, so this system depends on no other system's.

    An import of the ASR template's main would make every esp2_s2t recipe read
    that template's conf/, because main resolves the defaults from the package
    it is defined in.
    """
    from egs3.TEMPLATE.asr import run as generic
    from egs3.TEMPLATE.esp2_s2t import run as s2t

    assert s2t.main is not generic.main
    assert s2t.main.__module__ == "egs3.TEMPLATE.esp2_s2t.run"


def test_train_tokenizer_is_not_a_stage_this_template_offers():
    """An S2T vocabulary ships with the checkpoint, so there is none to train.

    The stage list is the only place this is decided: --stages accepts exactly
    these names, so a stage left out here cannot be reached at all.
    """
    from egs3.TEMPLATE.esp2_s2t import run as s2t

    assert "train_tokenizer" not in s2t.DEFAULT_STAGES
    parser = s2t.build_parser(stages=s2t.DEFAULT_STAGES)
    with pytest.raises(SystemExit):
        parser.parse_args(["--stages", "train_tokenizer"])


@pytest.mark.parametrize(
    "config_name",
    ["training.yaml", "inference.yaml", "metrics.yaml", "publication.yaml"],
)
def test_every_config_a_recipe_can_pass_has_a_default_here(config_name):
    """main() merges each of these, so each needs a default to merge over."""
    assert load_default_config(config_name, PACKAGE) is not None


def test_the_training_default_names_the_s2t_task_and_no_tokenizer():
    """The two places this template differs from a generic one."""
    cfg = load_default_config("training.yaml", PACKAGE)
    assert cfg.task == "espnet2.tasks.s2t.S2TTask"
    assert "tokenizer" not in cfg
    # The create_dataset stage prefers a recipe's DatasetBuilder; a default hook
    # would send a recipe that has one to a module it does not ship.
    assert cfg.create_dataset is None


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
