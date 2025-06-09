from argparse import Namespace

import numpy as np
import pytest

from espnet2.tasks.cls import CLSTask


def test_add_arguments():
    CLSTask.get_parser()


def test_add_arguments_help():
    parser = CLSTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        CLSTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        CLSTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        CLSTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        CLSTask.print_config(f)
    parser = CLSTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    retval = CLSTask.required_data_names(True, inference)
    assert "speech" in retval
    if not inference:
        assert "label" in retval


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    retval = CLSTask.optional_data_names(True, inference)
    assert "speech_lengths" in retval
    assert "label_lengths" in retval


def get_dummy_namespace():
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
    args = get_dummy_namespace()
    args.classification_type = "multi-label"
    model = CLSTask.build_model(args)
    args.classification_type = "multi-class"
    model = CLSTask.build_model(args)
