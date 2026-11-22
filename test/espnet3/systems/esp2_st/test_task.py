"""Tests for ESPnet3 ST task wiring."""

from argparse import Namespace

import pytest

from espnet3.systems.esp2_st.task import STTask


def test_add_arguments():
    """Ensure parser construction succeeds."""
    STTask.get_parser()


def test_add_arguments_help():
    """Ensure parser help exits cleanly."""
    parser = STTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Ensure main help exits cleanly."""
    with pytest.raises(SystemExit):
        STTask.main(cmd=["--help"])


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        STTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Ensure main without args exits with usage."""
    with pytest.raises(SystemExit):
        STTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """Ensure printed config can be parsed back."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        STTask.print_config(f)
    parser = STTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    """Speech is always required; the translation only outside inference."""
    retval = STTask.required_data_names(True, inference)

    assert "speech" in retval
    assert ("text" in retval) is not inference


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    """src_text feeds the auxiliary ASR decoder, so training only."""
    retval = STTask.optional_data_names(True, inference)

    assert retval == (() if inference else ("src_text",))


def test_build_model_rejects_invalid_token_list_type():
    """The target vocabulary must be a path or a list, as for ASR."""
    with pytest.raises(RuntimeError, match="token_list must be str or list"):
        STTask.build_model(_namespace(token_list=123))


def test_build_model_rejects_invalid_src_token_list_type():
    """The source side is checked the same way when it is given at all."""
    with pytest.raises(RuntimeError, match="token_list must be str or list"):
        STTask.build_model(_namespace(src_token_list=123))


def _namespace(**overrides):
    """A minimal ST namespace; `overrides` makes one field invalid."""
    args = Namespace(
        token_list=["<blank>", "<unk>", "a", "<sos/eos>"],
        src_token_list=None,
        model_conf={},
        input_size=80,
        frontend=None,
        frontend_conf={},
        specaug=None,
        specaug_conf={},
        normalize=None,
        normalize_conf={},
        preencoder=None,
        preencoder_conf={},
        encoder=None,
        encoder_conf={},
        postencoder=None,
        postencoder_conf={},
        decoder=None,
        decoder_conf={},
        extra_asr_decoder=None,
        extra_asr_decoder_conf={},
        extra_mt_decoder=None,
        extra_mt_decoder_conf={},
        md_encoder=None,
        md_encoder_conf={},
        hier_encoder=None,
        hier_encoder_conf={},
        extra_mt_encoder=None,
        extra_mt_encoder_conf={},
        ctc_conf={},
        src_ctc_conf={},
        init=None,
        model="espnet",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
