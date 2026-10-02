"""Tests for egs3/ami/esp2_s2t/conf/training.yaml.

These build the objects the config describes. Asserting the config's literals back at it
would pass whatever those literals said.
"""

import argparse

import ami_sot_paths
import pytest

pytest.importorskip("whisper")


def _config():
    """Load the config the way run.py does.

    conf/training.yaml is a partial overlay: the dataloader's iter_factory target,
    collate_fn and batch_bins all come from the shared template.
    load_config_with_defaults performs only Hydra `defaults:` composition, so a test
    using it would assert against an object the run stage never sees.
    """
    from espnet3.utils.config_utils import load_and_merge_config

    return load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / "training.yaml",
        config_name="training.yaml",
        default_package="egs3.TEMPLATE.asr",
        resolve=True,
    )


@pytest.mark.execution_timeout(120)
def test_the_preprocessor_the_config_names_actually_builds(tmp_path):
    """The preprocessor the config names actually builds.

    The three timestamp symbols default to OWSM's spellings and raise KeyError against a
    Whisper vocabulary.
    """
    import sys

    from hydra.utils import instantiate

    sys.path.insert(0, str(ami_sot_paths.RECIPE))
    options = dict(_config()["dataset"]["preprocessor"])
    # A path, not a list: Whisper BPE id 16259 is the literal string "???",
    # which OmegaConf reads as its MISSING marker, so a token list handed to
    # Hydra as a Python list raises MissingMandatoryValue. Production passes
    # ${token_list_path} for the same reason.
    options["token_list"] = str(_token_list_file(tmp_path))
    # DataOrganizer injects this at runtime, so the config omits it and a
    # direct instantiate has to supply it.
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
    """The model the config names resolves all five special symbols.

    A Whisper vocabulary has none of <blank>/<sos>/<eos>/<sop>/<na>, and ESPnetS2TModel
    resolves all five by token_list.index.
    """
    from espnet2.tasks.s2t import S2TTask

    config = _config()
    args = S2TTask.get_default_config()
    model_config = dict(config["model"])
    model_config.pop("token_list", None)
    # tiny keeps this inside the test budget; the symbols do not depend on size.
    model_config["encoder_conf"] = {"whisper_model": "tiny", "dropout_rate": 0.0}
    model_config["decoder_conf"] = {"whisper_model": "tiny"}
    args.update(token_list=_whisper_token_list(), **model_config)
    model = S2TTask.build_model(argparse.Namespace(**args))

    ids = [model.blank_id, model.sos, model.eos, model.sop, model.na]
    assert all(isinstance(i, int) for i in ids)
    assert len(set(ids)) == 5, ids


def test_the_separator_the_preprocessor_uses_is_the_one_the_corpus_was_built_with():
    """The most dangerous drift in this recipe.

    A token-list path mismatch fails loudly on a missing file. A separator mismatch
    trains silently against a symbol the corpus does not contain, and the two values
    live in different files: builder.separator in dataset/config.yaml,
    speaker_change_symbol in conf/training.yaml.
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
    """These are separate expressions in separate files and drifted once.

    The variable has to be set for this to mean anything: with it unset both expressions
    collapse to the current directory and the test would pass against the very mismatch
    it exists to catch.
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


# Every key espnet2/bin/s2t_inference.py indexes out of preprocessor_conf.
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
    """The saved config is the only channel between train and infer.

    save_espnet_config writes the model block to the root of <exp_dir>/config.yaml, and
    Speech2Text indexes the root-level preprocessor_conf directly, with no default.
    Leave it to S2TTask's defaults and decoding dies on OWSM's '<notimestamps>', which
    is not in a Whisper token list.
    """
    config = _config()
    assert config["model"]["preprocessor_conf"][key] == (
        config["dataset"]["preprocessor"][key]
    )


def test_the_recipe_carries_no_second_source_for_those_symbols():
    """They are interpolated, not copied.

    Two literal copies of the same symbol in one file is the drift this recipe keeps
    running into; the test above would still pass while the two said different things
    only if someone replaced the interpolation with a literal, so check the file text as
    well.
    """
    text = (ami_sot_paths.RECIPE / "conf" / "training.yaml").read_text()
    body = text.split("preprocessor_conf:", 1)[1].split("\n\n", 1)[0]
    for key in _READ_BY_INFERENCE:
        assert f"{key}: ${{dataset.preprocessor.{key}}}" in body, key
