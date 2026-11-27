"""Tests for egs3/ami/esp2_s2t/conf/training.yaml.

These tests build the objects that the config describes. A test that only compared
the config's literal values with themselves would pass for any values.
"""

import argparse

import ami_sot_paths
import pytest

pytest.importorskip("whisper")


def _config():
    """Load the config the way run.py does.

    conf/training.yaml is a partial overlay. The dataloader's iter_factory target,
    collate_fn and batch_bins all come from the shared template.
    load_config_with_defaults only does Hydra `defaults:` composition. A test that
    used it would check an object that the run stage never sees.
    """
    from espnet3.utils.config_utils import load_and_merge_config

    return load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / "training.yaml",
        config_name="training.yaml",
        default_package=ami_sot_paths.TEMPLATE_PACKAGE,
        resolve=True,
    )


def test_the_create_dataset_stage_can_read_this_config():
    """The create_dataset stage can read this config's create_dataset block.

    The test uses the expression from espnet3/systems/base/system.py,
    ``dict(getattr(config, "create_dataset", OmegaConf.create({})))``. The default
    is used only when the key is absent. A key that is present and null is passed
    to ``dict``, which raises TypeError before the stage reaches the recipe's
    DatasetBuilder.
    """
    from omegaconf import OmegaConf

    config = _config()
    block = getattr(config, "create_dataset", OmegaConf.create({}))
    assert dict(block) is not None


@pytest.mark.execution_timeout(120)
def test_the_preprocessor_the_config_names_actually_builds(tmp_path):
    """The preprocessor in the config can be built with a Whisper token list.

    The three timestamp symbols default to OWSM's spellings, which raise KeyError
    with a Whisper vocabulary.
    """
    import sys

    from hydra.utils import instantiate

    sys.path.insert(0, str(ami_sot_paths.RECIPE))
    options = dict(_config()["dataset"]["preprocessor"])
    # The token list is passed as a file path. Whisper BPE id 16259 is the literal
    # string "???", which OmegaConf reads as its MISSING marker. A token list given
    # to Hydra as a Python list therefore raises MissingMandatoryValue. Production
    # passes ${token_list_path} for the same reason.
    options["token_list"] = str(_token_list_file(tmp_path))
    # DataOrganizer adds this value at runtime, so the config does not have it. A
    # direct instantiate call must pass it.
    options["train"] = False
    preprocessor = instantiate(options, _convert_="all")
    assert preprocessor.speaker_change_id == 25629


def _token_list_file(tmp_path):
    """Write the vocabulary to a file and return its path."""
    target = tmp_path / "tokens.txt"
    target.write_text("\n".join(_whisper_token_list()) + "\n")
    return target


def _whisper_token_list():
    from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

    vocab = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", "en", task="transcribe"
    ).tokenizer.tokenizer.get_vocab()
    tokens = [None] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    return tokens


@pytest.mark.execution_timeout(600)
def test_the_model_the_config_names_resolves_all_five_special_symbols():
    """The configured model gets a distinct id for all five special symbols.

    A Whisper vocabulary has none of <blank>/<sos>/<eos>/<sop>/<na>, and
    ESPnetS2TModel looks up all five with token_list.index.
    """
    from espnet2.tasks.s2t import S2TTask

    config = _config()
    args = S2TTask.get_default_config()
    model_config = dict(config["model"])
    model_config.pop("token_list", None)
    # The tiny model keeps this test inside its time budget. The symbols do not
    # depend on the model size.
    model_config["encoder_conf"] = {"whisper_model": "tiny", "dropout_rate": 0.0}
    model_config["decoder_conf"] = {"whisper_model": "tiny"}
    args.update(token_list=_whisper_token_list(), **model_config)
    model = S2TTask.build_model(argparse.Namespace(**args))

    ids = [model.blank_id, model.sos, model.eos, model.sop, model.na]
    assert all(isinstance(i, int) for i in ids)
    assert len(set(ids)) == 5, ids


def test_the_separator_the_preprocessor_uses_is_the_one_the_corpus_was_built_with():
    """The preprocessor uses the separator that the corpus was built with.

    A wrong token-list path raises an error for the missing file. A wrong separator
    raises no error, and the model trains with a symbol that the corpus does not
    contain. The two values are in different files. builder.separator is in
    dataset/config.yaml, and speaker_change_symbol is in conf/training.yaml.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ami_s2t_builder_for_sep", ami_sot_paths.RECIPE / "dataset" / "builder.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    config = _config()
    assert (
        config["dataset"]["preprocessor"]["speaker_change_symbol"]
        == builder._CONFIG["separator"]
    )


def test_the_token_list_the_config_reads_is_the_one_the_builder_writes(monkeypatch):
    """The config reads the token list from the path that the builder writes to.

    The two paths are separate expressions in separate files, and they once
    differed. The test sets AMI_SOT_DATA_ROOT. If it were unset, both expressions
    would resolve to the current directory, and the test would pass even with the
    mismatch that it is meant to catch.
    """
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", "/nonexistent-root-for-this-test")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ami_s2t_builder_for_config", ami_sot_paths.RECIPE / "dataset" / "builder.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    written = (
        Path(builder._CONFIG["data_root"]) / builder._CONFIG["token_list"]
    ).resolve()
    config = _config()
    read = Path(config["model"]["token_list"]).resolve()
    assert written == read


# Every key that espnet2/bin/s2t_inference.py reads from preprocessor_conf.
_READ_BY_INFERENCE = (
    "notime_symbol",
    "first_time_symbol",
    "last_time_symbol",
    "fs",
    "speech_length",
    "speech_resolution",
)


@pytest.mark.parametrize("key", _READ_BY_INFERENCE)
def test_the_checkpoint_hands_inference_the_symbols_the_recipe_trained_with(key):
    """model.preprocessor_conf has the same symbols as dataset.preprocessor.

    The saved config is the only link between train and infer. save_espnet_config
    writes the model block to the root of <exp_dir>/config.yaml. Speech2Text reads
    the root-level preprocessor_conf directly, with no default. With S2TTask's
    defaults, decoding fails on OWSM's '<notimestamps>', which is not in a Whisper
    token list.
    """
    config = _config()
    assert config["model"]["preprocessor_conf"][key] == (
        config["dataset"]["preprocessor"][key]
    )


def test_the_recipe_carries_no_second_source_for_those_symbols():
    """The preprocessor_conf values are interpolated from dataset.preprocessor.

    This recipe has had two literal copies of one symbol in the same file more than
    once. If someone replaced the interpolation with a literal copy, the test above
    would pass until the two copies differ. So this test also checks the file text.
    """
    text = (ami_sot_paths.RECIPE / "conf" / "training.yaml").read_text()
    body = text.split("preprocessor_conf:", 1)[1].split("\n\n", 1)[0]
    for key in _READ_BY_INFERENCE:
        assert f"{key}: ${{dataset.preprocessor.{key}}}" in body, key


def test_one_variable_sets_the_separator_everywhere(monkeypatch):
    """AMI_SOT_SPEAKER_CHANGE_SYMBOL sets the symbol in every config.

    The corpus, training, inference and metrics configs must use the same
    symbol. If one of them used another symbol, the data, the token list and
    the decoder would not match, and no error would be raised.
    """
    from espnet3.utils.config_utils import (
        load_and_merge_config,
        load_config_with_defaults,
    )

    monkeypatch.setenv("AMI_SOT_SPEAKER_CHANGE_SYMBOL", "<sc>")

    def load(name):
        return load_and_merge_config(
            ami_sot_paths.RECIPE / "conf" / name,
            config_name=name,
            default_package=ami_sot_paths.TEMPLATE_PACKAGE,
            resolve=True,
        )

    dataset = load_config_with_defaults(
        str(ami_sot_paths.RECIPE / "dataset" / "config.yaml")
    )
    found = [
        dataset["builder"]["separator"],
        load("training.yaml").dataset.preprocessor.speaker_change_symbol,
        load("inference.yaml").model.speaker_change_symbol,
        *[entry.metric.speaker_change_symbol for entry in load("metrics.yaml").metrics],
    ]
    assert found == ["<sc>"] * len(found)
