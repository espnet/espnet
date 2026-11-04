"""Tests for espnet3/systems/esp2_s2t/src/packing.py."""

import logging

import yaml

from espnet3.systems.esp2_s2t.src.packing import inline_token_list


def _bundle(tmp_path, token_list, train_config="${recipe_dir}/exp/run/config.yaml"):
    """A bundle as pack_model leaves it, with a train config naming token_list."""
    bundle = tmp_path / "bundle"
    (bundle / "conf").mkdir(parents=True)
    (bundle / "exp" / "run").mkdir(parents=True)
    (bundle / "conf" / "inference.yaml").write_text(
        yaml.safe_dump({"recipe_dir": ".", "model": {"s2t_train_config": train_config}})
    )
    config = bundle / "exp" / "run" / "config.yaml"
    config.write_text(yaml.safe_dump({"token_list": token_list, "frontend": None}))
    return bundle, config


def test_a_token_list_path_becomes_the_list_espnet2_would_read(tmp_path):
    """The bundle must not need the corpus the checkpoint was trained on.

    The list is read as S2TTask.build_model reads the file, trailing
    whitespace stripped, so the model is sized and indexed exactly as before.
    """
    tokens = tmp_path / "corpus" / "tokens.txt"
    tokens.parent.mkdir()
    tokens.write_text("<blank>\n t \n????\n", encoding="utf-8")
    bundle, config = _bundle(tmp_path, str(tokens))

    inline_token_list(bundle)

    args = yaml.safe_load(config.read_text())
    assert args == {"token_list": ["<blank>", " t", "????"], "frontend": None}


def test_a_token_list_already_written_out_is_left_alone(tmp_path):
    bundle, config = _bundle(tmp_path, ["<blank>", "a"])
    before = config.read_text()
    inline_token_list(bundle)
    assert config.read_text() == before


def test_a_train_config_outside_the_bundle_is_not_touched(tmp_path, caplog):
    """It is not the bundle's to rewrite, and the bundle will not be portable."""
    outside = tmp_path / "elsewhere" / "config.yaml"
    outside.parent.mkdir()
    outside.write_text(yaml.safe_dump({"token_list": "/corpus/tokens.txt"}))
    bundle, _ = _bundle(tmp_path, "/corpus/tokens.txt", train_config=str(outside))

    with caplog.at_level(logging.WARNING):
        inline_token_list(bundle)

    assert yaml.safe_load(outside.read_text()) == {"token_list": "/corpus/tokens.txt"}
    assert "not in the bundle" in caplog.text
