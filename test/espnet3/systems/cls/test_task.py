"""Tests for ESPnet3 CLS task wiring.

`espnet3/systems/cls/task.py` is a copy of `espnet2/tasks/cls.py` with three
changes, all in `build_model`. The tests are grouped accordingly: the ESPnet2
ones are kept so the copy cannot drift unnoticed, then the changes, then the
`build_model` branches neither file covers.
"""

from argparse import Namespace

import numpy as np
import pytest

from espnet3.systems.cls.espnet_model import ClassificationModel
from espnet3.systems.cls.task import CLSTask

# ===============================================================
# Test Case Summary
# ===============================================================
#
# Copied from test/espnet2/tasks/test_cls.py
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_add_arguments                          | Parser construction succeeds.|
# | test_add_arguments_help                     | --help exits cleanly.        |
# | test_main_help                              | main --help exits cleanly.   |
# | test_main_print_config                      | --print_config exits cleanly.|
# | test_main_with_no_args                      | No args exits with usage.    |
# | test_print_config_and_load_it               | The printed config parses    |
# |                                             | back.                        |
# | test_required_data_names                    | speech/label in training,    |
# |                                             | speech alone at inference.   |
# | test_optional_data_names                    | Length fields are optional.  |
# | test_build_preprocess_fn                    | The preprocessor tokenizes   |
# |                                             | the label column.            |
# | test_build_collate_fn                       | Floats pad with 0.0 and ints |
# |                                             | with -1.                     |
# | test_build_model                            | Both classification types    |
# |                                             | build.                       |
#
# What ESPnet3 changes in build_model
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_model_returns_espnet3_model      | model_choices resolves       |
# |                                             | `espnet` to the ESPnet3      |
# |                                             | model.                       |
# | test_build_model_rejects_an_unknown_model_name | An unknown name is        |
# |                                             | rejected by the registry.    |
# | test_build_model_defaults_the_model_name    | An unset name falls back to  |
# |                                             | the registry default.        |
# | test_build_model_rejects_a_null_model_name  | An explicit null is an       |
# |                                             | error, not a fallback.       |
# | test_build_model_forwards_freeze_param      | freeze_param reaches the     |
# |                                             | model and freezes a subtree. |
#
# Branches the ESPnet2 tests leave uncovered
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_preprocess_fn_can_be_disabled    | use_preprocessor: false      |
# |                                             | returns None.                |
# | test_build_model_inlines_token_list_file    | A path is read and written   |
# |                          | back into args.token_list as a list.            |
# | test_build_model_rejects_invalid_token_list_type | Neither str nor list    |
# |                          | raises RuntimeError. ASRTask's ESPnet3 test      |
# |                          | file adds the same case.                        |
# | test_build_model_excludes_unk_from_classes  | n_classes is len(token_list) |
# |                          | - 1; the metrics drop the same trailing entry.  |
# | test_build_model_with_the_optional_blocks   | frontend / normalize /       |
# |                          | preencoder are built and chained by size.       |
# | test_build_model_without_the_optional_blocks | All of them stay None.      |


# ---------------------------------------------------------------
# Copied from test/espnet2/tasks/test_cls.py
# ---------------------------------------------------------------


def test_add_arguments():
    """Ensure parser construction succeeds."""
    CLSTask.get_parser()


def test_add_arguments_help():
    """Ensure parser help exits cleanly."""
    parser = CLSTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Ensure main help exits cleanly."""
    with pytest.raises(SystemExit):
        CLSTask.main(cmd=["--help"])


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        CLSTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Ensure main without args exits with usage."""
    with pytest.raises(SystemExit):
        CLSTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """Ensure printed config can be parsed back."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        CLSTask.print_config(f)
    parser = CLSTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    """Ensure the label field is required only outside inference."""
    retval = CLSTask.required_data_names(True, inference)
    assert "speech" in retval
    if not inference:
        assert "label" in retval


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    """Ensure the length fields stay optional."""
    retval = CLSTask.optional_data_names(True, inference)
    assert "speech_lengths" in retval
    assert "label_lengths" in retval


def get_dummy_namespace():
    """Build a minimal CLS config namespace for model and preprocess tests."""
    return Namespace(
        token_type="word",
        token_list=["class1", "class2", "class3", "class4", "<unk>"],
        classification_type="multi-class",
        input_size=40,
        frontend="frontend",
        frontend_conf={"n_fft": 51, "win_length": 40, "hop_length": 16},
        specaug="specaug",
        specaug_conf={"apply_time_warp": True, "time_mask_width_range": 4},
        normalize=None,
        normalize_conf=None,
        preencoder=None,
        encoder="transformer",
        encoder_conf={
            "output_size": 40,
            "linear_units": 4,
            "num_blocks": 2,
        },
        decoder="linear",
        decoder_conf={
            "pooling": "mean",
        },
        init="normal",
        model_conf={},
    )


def test_build_preprocess_fn():
    """Ensure the preprocessor maps label strings to token ids."""
    args = get_dummy_namespace()

    preprocessor_args = {
        "use_preprocessor": True,
        "non_linguistic_symbols": None,
        "cleaner": None,
        "g2p": None,
        "use_lang_prompt": False,
        "use_nlp_prompt": False,
    }
    args.__dict__.update(preprocessor_args)

    preprocess = CLSTask.build_preprocess_fn(args, True)
    data = {}
    data["label"] = "class4 class2"
    data_preprocessed = preprocess._text_process(data)
    assert "label" in data_preprocessed
    assert np.all(data_preprocessed["label"] == np.array([3, 1]))


def test_build_collate_fn():
    """Ensure floats pad with 0.0 and integer labels pad with -1."""
    args = get_dummy_namespace()
    collate_fn = CLSTask.build_collate_fn(args, True)
    # Following test is same as espnet/test/espnet2/train/test_collate_fn.py:test_
    float_pad_value = 0.0
    int_pad_value = -1
    data = [
        ("id", dict(a=np.random.randn(3, 5), b=np.random.randn(4).astype(np.int64))),
        ("id2", dict(a=np.random.randn(2, 5), b=np.random.randn(3).astype(np.int64))),
    ]
    t = collate_fn(data)

    desired = dict(
        a=np.stack(
            [
                data[0][1]["a"],
                np.pad(
                    data[1][1]["a"],
                    [(0, 1), (0, 0)],
                    mode="constant",
                    constant_values=float_pad_value,
                ),
            ]
        ),
        b=np.stack(
            [
                data[0][1]["b"],
                np.pad(
                    data[1][1]["b"],
                    [(0, 1)],
                    mode="constant",
                    constant_values=int_pad_value,
                ),
            ]
        ),
        a_lengths=np.array([3, 2], dtype=np.int64),
        b_lengths=np.array([4, 3], dtype=np.int64),
    )

    np.testing.assert_array_equal(t[1]["a"], desired["a"])
    np.testing.assert_array_equal(t[1]["b"], desired["b"])
    np.testing.assert_array_equal(t[1]["a_lengths"], desired["a_lengths"])
    np.testing.assert_array_equal(t[1]["b_lengths"], desired["b_lengths"])


def test_build_model():
    """Ensure both classification types build."""
    args = get_dummy_namespace()
    args.classification_type = "multi-label"
    _ = CLSTask.build_model(args)
    args.classification_type = "multi-class"
    _ = CLSTask.build_model(args)


# ---------------------------------------------------------------
# The three ESPnet3 changes
# ---------------------------------------------------------------


def test_build_model_returns_espnet3_model():
    """Ensure `model_choices` resolves `espnet` to the ESPnet3 model."""
    model = CLSTask.build_model(get_dummy_namespace())

    assert isinstance(model, ClassificationModel)


def test_build_model_rejects_an_unknown_model_name():
    """Ensure an unknown `model` name is rejected.

    A hardcoded model class would ignore the name and build anyway.
    """
    args = get_dummy_namespace()
    args.model = "bogus"

    with pytest.raises(ValueError, match="--model must be one of"):
        CLSTask.build_model(args)


def test_build_model_defaults_the_model_name():
    """Ensure an unset `model` name falls back to the registry default."""
    args = get_dummy_namespace()
    assert not hasattr(args, "model")

    assert isinstance(CLSTask.build_model(args), ClassificationModel)


def test_build_model_rejects_a_null_model_name():
    """Ensure `model: null` is an error rather than a silent fallback."""
    args = get_dummy_namespace()
    args.model = None

    with pytest.raises(TypeError):
        CLSTask.build_model(args)


def test_build_model_forwards_freeze_param():
    """Ensure `freeze_param` reaches the model and freezes the named subtree."""
    args = get_dummy_namespace()
    args.freeze_param = ["encoder"]

    model = CLSTask.build_model(args)

    assert model.freeze_param == ["encoder"]
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())


# ---------------------------------------------------------------
# Branches the ESPnet2 tests leave uncovered
# ---------------------------------------------------------------


def test_build_preprocess_fn_can_be_disabled():
    """Ensure use_preprocessor: false skips preprocessing entirely."""
    assert CLSTask.build_preprocess_fn(Namespace(use_preprocessor=False), True) is None


def test_build_model_inlines_token_list_file(tmp_path):
    """Ensure a token_list path is read and replaced by its contents."""
    token_list = tmp_path / "token_list"
    token_list.write_text("class1\nclass2\n<unk>\n", encoding="utf-8")
    args = get_dummy_namespace()
    args.token_list = str(token_list)

    model = CLSTask.build_model(args)

    # ESPnet2 calls this keeping the model "portable": the saved config must
    # not refer back to a file on the training machine.
    assert args.token_list == ["class1", "class2", "<unk>"]
    assert model.token_list == ["class1", "class2", "<unk>"]


def test_build_model_rejects_invalid_token_list_type():
    """Ensure a token_list that is neither a path nor a list is refused."""
    args = get_dummy_namespace()
    args.token_list = 123

    with pytest.raises(RuntimeError, match="token_list must be str or list"):
        CLSTask.build_model(args)


def test_build_model_excludes_unk_from_classes():
    """Ensure <unk> is counted out of the classifier's output dimension.

    The classification metrics drop the same trailing entry, so the class
    count has to agree with them.
    """
    model = CLSTask.build_model(get_dummy_namespace())

    assert model.vocab_size == 4  # five tokens, minus <unk>
    assert model.decoder.output_size() == 4


def test_build_model_with_the_optional_blocks():
    """Ensure frontend, normalize and preencoder are built and chained by size.

    With `input_size` unset the frontend decides the feature width, and the
    preencoder decides what the encoder receives.
    """
    args = get_dummy_namespace()
    args.input_size = None
    args.frontend = "default"
    args.frontend_conf = {
        "n_fft": 128,
        "win_length": 128,
        "hop_length": 64,
        "n_mels": 20,
    }
    args.normalize = "utterance_mvn"
    args.normalize_conf = {}
    args.preencoder = "linear"
    args.preencoder_conf = {"input_size": 20, "output_size": 8}

    model = CLSTask.build_model(args)

    assert model.frontend.output_size() == 20  # what the preencoder is given
    assert model.preencoder.output_size() == 8  # what the encoder is given
    assert model.normalize is not None
    assert model.specaug is not None


def test_build_model_without_the_optional_blocks():
    """Ensure every optional block stays None when it is not configured."""
    args = get_dummy_namespace()
    args.specaug = None

    model = CLSTask.build_model(args)

    assert model.frontend is None  # input_size is set, so none is built
    assert model.specaug is None
    assert model.normalize is None
    assert model.preencoder is None
