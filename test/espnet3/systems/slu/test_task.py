"""Tests for ESPnet3 SLU task wiring."""

from argparse import Namespace

import pytest

from espnet3.systems.slu.task import SLUTask


def test_add_arguments():
    """Ensure parser construction succeeds."""
    SLUTask.get_parser()


def test_add_arguments_help():
    """Ensure parser help exits cleanly."""
    parser = SLUTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Ensure main help exits cleanly."""
    with pytest.raises(SystemExit):
        SLUTask.main(cmd=["--help"])


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        SLUTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Ensure main without args exits with usage."""
    with pytest.raises(SystemExit):
        SLUTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """Ensure printed config can be parsed back."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        SLUTask.print_config(f)
    parser = SLUTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names_training(inference):
    """speech and text required during training; only speech during inference."""
    retval = SLUTask.required_data_names(train=True, inference=inference)
    if not inference:
        assert "speech" in retval
        assert "text" in retval
    else:
        assert "speech" in retval
        assert "text" not in retval


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names_includes_transcript(inference):
    """transcript is always in optional data names."""
    retval = SLUTask.optional_data_names(train=True, inference=inference)
    assert "transcript" in retval


def _make_preprocess_args(use_preprocessor=True, transcript_token_list=None):
    """Build a minimal Namespace for build_preprocess_fn."""
    return Namespace(
        use_preprocessor=use_preprocessor,
        token_type="char",
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        transcript_token_list=transcript_token_list,
        bpemodel=None,
        non_linguistic_symbols=None,
        cleaner=None,
        g2p=None,
        speech_volume_normalize=None,
        rir_scp=None,
        rir_apply_prob=1.0,
        noise_scp=None,
        noise_apply_prob=1.0,
        noise_db_range="13_15",
        short_noise_thres=0.5,
    )


def test_build_preprocess_fn_returns_none_when_disabled():
    """build_preprocess_fn returns None when use_preprocessor=False."""
    args = _make_preprocess_args(use_preprocessor=False)
    result = SLUTask.build_preprocess_fn(args, train=True)
    assert result is None


def test_build_preprocess_fn_returns_preprocessor():
    """build_preprocess_fn returns an SLUPreprocessor when enabled."""
    from espnet2.train.preprocessor import SLUPreprocessor

    args = _make_preprocess_args(use_preprocessor=True)
    result = SLUTask.build_preprocess_fn(args, train=True)
    assert isinstance(result, SLUPreprocessor)


def test_build_preprocess_fn_with_transcript_token_list():
    """build_preprocess_fn accepts a separate transcript_token_list."""
    from espnet2.train.preprocessor import SLUPreprocessor

    args = _make_preprocess_args(
        transcript_token_list=["<blank>", "<unk>", "x", "y", "<eos>"]
    )
    result = SLUTask.build_preprocess_fn(args, train=True)
    assert isinstance(result, SLUPreprocessor)


def _make_build_model_args(
    transcript_token_list=None, two_pass=False, encoder="transformer"
):
    """Build a minimal Namespace for build_model."""
    return Namespace(
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        transcript_token_list=transcript_token_list,
        two_pass=two_pass,
        pre_postencoder_norm=False,
        input_size=20,
        frontend=None,
        frontend_conf={},
        specaug=None,
        specaug_conf={},
        normalize=None,
        normalize_conf={},
        preencoder=None,
        preencoder_conf={},
        encoder=encoder,
        encoder_conf={"output_size": 20, "linear_units": 4, "num_blocks": 2},
        postencoder=None,
        postencoder_conf={},
        deliberationencoder=None,
        deliberationencoder_conf={},
        postdecoder=None,
        postdecoder_conf={},
        decoder="transformer",
        decoder_conf={"linear_units": 4, "num_blocks": 2},
        ctc_conf={},
        joint_net_conf=None,
        model="espnet",
        model_conf={},
        init=None,
    )


def test_build_model_basic():
    """build_model constructs a minimal ESPnetSLUModel."""
    from espnet2.slu.espnet_model import ESPnetSLUModel

    args = _make_build_model_args()
    model = SLUTask.build_model(args)
    assert isinstance(model, ESPnetSLUModel)


def test_build_model_with_conformer_encoder():
    """build_model works with a conformer encoder."""
    from espnet2.slu.espnet_model import ESPnetSLUModel

    args = _make_build_model_args(encoder="conformer")
    model = SLUTask.build_model(args)
    assert isinstance(model, ESPnetSLUModel)


def test_build_model_with_transcript_token_list():
    """build_model handles a separate transcript_token_list."""
    from espnet2.slu.espnet_model import ESPnetSLUModel

    args = _make_build_model_args(
        transcript_token_list=["<blank>", "<unk>", "x", "y", "<eos>"]
    )
    model = SLUTask.build_model(args)
    assert isinstance(model, ESPnetSLUModel)


def test_build_model_with_two_pass():
    """build_model propagates two_pass into model_conf."""
    from espnet2.slu.espnet_model import ESPnetSLUModel

    args = _make_build_model_args(
        transcript_token_list=["<blank>", "<unk>", "x", "y", "<eos>"],
        two_pass=True,
    )
    model = SLUTask.build_model(args)
    assert isinstance(model, ESPnetSLUModel)


def test_build_model_invalid_token_list():
    """build_model raises RuntimeError for an unsupported token_list type."""
    args = _make_build_model_args()
    args.token_list = 12345  # invalid type
    with pytest.raises(RuntimeError):
        SLUTask.build_model(args)


def test_build_model_with_postencoder():
    """build_model constructs model with a transformer postencoder."""
    from espnet2.slu.espnet_model import ESPnetSLUModel

    args = _make_build_model_args()
    args.postencoder = "transformer"
    args.postencoder_conf = {"output_size": 20, "linear_units": 4}
    model = SLUTask.build_model(args)
    assert isinstance(model, ESPnetSLUModel)
    assert model.postencoder is not None


def test_build_model_with_deliberation_encoder():
    """build_model constructs model with a conformer deliberation encoder."""
    from espnet2.slu.espnet_model import ESPnetSLUModel

    args = _make_build_model_args()
    args.deliberationencoder = "conformer"
    args.deliberationencoder_conf = {"output_size": 20, "linear_units": 4}
    model = SLUTask.build_model(args)
    assert isinstance(model, ESPnetSLUModel)
    assert model.deliberationencoder is not None
