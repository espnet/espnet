from argparse import Namespace

import pytest

from espnet2.tasks.asr_transducer import ASRTransducerTask


def test_add_arguments():
    ASRTransducerTask.get_parser()


def test_add_arguments_help():
    parser = ASRTransducerTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        ASRTransducerTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        ASRTransducerTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        ASRTransducerTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        ASRTransducerTask.print_config(f)
    parser = ASRTransducerTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


def get_dummy_namespace():
    return Namespace(
        token_type="char",
        token_list=["<blank>", "a", "b", "c", "<space>", "<unk>"],
        input_size=40,
        frontend="frontend",
        frontend_conf={"n_fft": 51, "win_length": 40, "hop_length": 16},
        specaug="specaug",
        specaug_conf={"apply_time_warp": True, "time_mask_width_range": 4},
        normalize=None,
        normalize_conf=None,
        encoder_conf={
            "body_conf": [
                {
                    "block_type": "conformer",
                    "hidden_size": 16,
                    "linear_size": 8,
                    "conv_mod_kernel_size": 3,
                }
            ]
        },
        decoder="stateless",
        decoder_conf={"embed_size": 8},
        joint_network_conf={"joint_space_size": 8},
        model_conf={"transducer_weight": 1.0},
        init=None,
    )


def test_build_model():
    args = get_dummy_namespace()

    _ = ASRTransducerTask.build_model(args)

    with pytest.raises(RuntimeError):
        args.token_list = -1

        _ = ASRTransducerTask.build_model(args)


def test_build_collate_fn():
    args = get_dummy_namespace()

    _ = ASRTransducerTask.build_collate_fn(args, True)


@pytest.mark.parametrize("use_preprocessor", [True, False])
def test_build_preprocess_fn(use_preprocessor):
    args = get_dummy_namespace()

    new_args = {
        "use_preprocessor": use_preprocessor,
        "bpemodel": None,
        "non_linguistic_symbols": args.token_list,
        "cleaner": None,
        "g2p": None,
    }
    args.__dict__.update(new_args)

    _ = ASRTransducerTask.build_preprocess_fn(args, True)


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    retval = ASRTransducerTask.required_data_names(True, inference)

    if inference:
        assert retval == ("speech",)
    else:
        assert retval == ("speech", "text")


def test_optional_data_names():
    retval = ASRTransducerTask.optional_data_names(True, True)

    assert not retval
