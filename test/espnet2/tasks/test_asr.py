from argparse import Namespace

import pytest

from espnet2.tasks.asr import ASRTask


def test_add_arguments():
    ASRTask.get_parser()


def test_add_arguments_help():
    parser = ASRTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        ASRTask.print_config(f)
    parser = ASRTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    retval = ASRTask.optional_data_names(True, inference)

    assert "prompt" in retval


def get_dummy_namespace():
    pytest.importorskip("whisper")
    return Namespace(
        token_type="whisper_multilingual",
        token_list=["<blank>", "a", "b", "c", "<space>", "<unk>"],
        input_size=40,
        frontend="frontend",
        frontend_conf={"n_fft": 51, "win_length": 40, "hop_length": 16},
        specaug="specaug",
        specaug_conf={"apply_time_warp": True, "time_mask_width_range": 4},
        normalize=None,
        normalize_conf=None,
        encoder="transformer",
        encoder_conf={
            "output_size": 40,
            "linear_units": 4,
            "num_blocks": 2,
        },
        decoder="transformer",
        decoder_conf={
            "linear_units": 4,
            "num_blocks": 2,
        },
        init=None,
        ctc_conf={},
        model_conf={},
    )


def test_build_preprocess_fn_prompt():
    pytest.importorskip("whisper")
    args = get_dummy_namespace()

    new_args = {
        "use_preprocessor": True,
        "bpemodel": "whisper_multilingual",
        "non_linguistic_symbols": None,
        "cleaner": None,
        "g2p": None,
        "use_lang_prompt": True,
        "use_nlp_prompt": False,
    }
    args.__dict__.update(new_args)

    preprocess = ASRTask.build_preprocess_fn(args, True)
    data = {}
    data["text"] = "a a"
    data["prompt"] = "a"
    preprocess._text_process(data)


def test_build_preprocess_fn_nlp_prompt():
    pytest.importorskip("whisper")
    args = get_dummy_namespace()

    new_args = {
        "use_preprocessor": True,
        "bpemodel": "whisper_multilingual",
        "non_linguistic_symbols": None,
        "cleaner": None,
        "g2p": None,
        "use_lang_prompt": False,
        "use_nlp_prompt": True,
    }
    args.__dict__.update(new_args)

    preprocess = ASRTask.build_preprocess_fn(args, True)
    data = {}
    data["text"] = "a a"
    data["prompt"] = "a a"
    preprocess._text_process(data)
