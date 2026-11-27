"""Tests for the S2T task of the esp2_s2t system."""

from argparse import Namespace

import pytest

from espnet2.s2t.espnet_model import ESPnetS2TModel
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import S2TPreprocessor
from espnet3.systems.esp2_s2t.task import S2TTask

# ESPnetS2TModel looks up <blank>, <sos>, <eos>, <sop> and <na> in this list, and
# S2TPreprocessor looks up <notimestamps>, <0.00> and <30.00>.
_TOKENS = [
    "<blank>",
    "<unk>",
    "<na>",
    "<sop>",
    "<notimestamps>",
    "<0.00>",
    "<30.00>",
    "a",
    "b",
    "<sos>",
    "<eos>",
]


def test_add_arguments():
    """The parser can be built."""
    S2TTask.get_parser()


def test_add_arguments_help():
    """The parser prints its help and exits."""
    parser = S2TTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Calling main with --help prints the help and exits."""
    with pytest.raises(SystemExit):
        S2TTask.main(cmd=["--help"])


def test_main_print_config():
    """Calling main with --print_config prints the default config and exits."""
    with pytest.raises(SystemExit):
        S2TTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Calling main without arguments prints the usage and exits."""
    with pytest.raises(SystemExit):
        S2TTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """The printed config can be parsed again."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        S2TTask.print_config(f)
    parser = S2TTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    """Speech is always required, and the text only outside inference."""
    retval = S2TTask.required_data_names(True, inference)

    assert "speech" in retval
    assert ("text" in retval) is not inference


def test_optional_data_names():
    """The previous text, the CTC text and the texts of speakers 2 to 4 are optional."""
    assert S2TTask.optional_data_names() == (
        "text_prev",
        "text_ctc",
        "text_spk2",
        "text_spk3",
        "text_spk4",
    )


def test_build_collate_fn_pads_token_ids_with_minus_one():
    """Token ids are padded with -1, because id 0 is the CTC blank."""
    collate = S2TTask.build_collate_fn(_namespace(), train=True)

    assert isinstance(collate, CommonCollateFn)
    assert collate.int_pad_value == -1
    assert collate.float_pad_value == 0.0


def test_build_preprocess_fn_returns_none_without_preprocessor():
    """No preprocessing function is built when use_preprocessor is false."""
    args = _namespace(use_preprocessor=False)

    assert S2TTask.build_preprocess_fn(args, train=True) is None


def test_build_preprocess_fn_builds_the_s2t_preprocessor():
    """The s2t preprocessor is built from the token settings of the namespace."""
    preprocessor = S2TTask.build_preprocess_fn(_namespace(), train=True)

    assert isinstance(preprocessor, S2TPreprocessor)


def test_build_model_rejects_an_invalid_token_list_type():
    """The token list must be a path or a list."""
    with pytest.raises(RuntimeError, match="token_list must be str or list"):
        S2TTask.build_model(_namespace(token_list=123))


def test_build_model_builds_a_small_model_from_a_token_list_file(tmp_path):
    """A model with one encoder block and one decoder block can be built."""
    token_list = tmp_path / "tokens.txt"
    token_list.write_text("\n".join(_TOKENS) + "\n")
    args = _namespace(token_list=str(token_list))

    model = S2TTask.build_model(args)

    assert isinstance(model, ESPnetS2TModel)
    assert model.vocab_size == len(_TOKENS)
    # The path is replaced by the list, so the saved config needs no file.
    assert args.token_list == _TOKENS


def _namespace(**overrides):
    """Return the arguments of a small S2T model, with ``overrides`` applied."""
    args = Namespace(
        token_list=list(_TOKENS),
        token_type="word",
        bpemodel=None,
        cleaner=None,
        g2p=None,
        non_linguistic_symbols=None,
        use_preprocessor=True,
        preprocessor="s2t",
        preprocessor_conf={},
        input_size=8,
        frontend=None,
        frontend_conf={},
        specaug=None,
        specaug_conf={},
        normalize=None,
        normalize_conf={},
        preencoder=None,
        preencoder_conf={},
        encoder="transformer",
        encoder_conf=dict(
            output_size=8,
            attention_heads=2,
            linear_units=8,
            num_blocks=1,
            input_layer="linear",
        ),
        postencoder=None,
        postencoder_conf={},
        decoder="transformer",
        decoder_conf=dict(attention_heads=2, linear_units=8, num_blocks=1),
        ctc_conf={},
        model="espnet",
        model_conf={},
        init=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
