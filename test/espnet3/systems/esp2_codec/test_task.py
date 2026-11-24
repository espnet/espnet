"""Tests for the ESPnet3 GAN codec task compatibility copy."""

import argparse
import sys
import types

import numpy as np
import pytest
import torch

from espnet2.gan_codec.encodec.encodec import Encodec
from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
from espnet2.tasks.gan_codec import GANCodecTask as ESPnet2GANCodecTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet3.systems.esp2_codec.task import GANCodecTask, codec_choices

# A reduced EnCodec, the model of egs3/mini_an4/esp2_codec/conf/training.yaml.
TINY_ENCODEC_CONF = dict(
    sampling_rate=16000,
    generator_params=dict(
        hidden_dim=8,
        encdec_channels=1,
        encdec_n_filters=2,
        encdec_n_residual_layers=1,
        encdec_ratios=[4, 2],
        encdec_lstm=1,
        quantizer_n_q=2,
        quantizer_bins=4,
        quantizer_kmeans_init=False,
        quantizer_target_bandwidth=[0.5, 1],
    ),
    discriminator_params=dict(
        msstft_discriminator_params=dict(
            filters=2,
            n_ffts=[64, 32],
            hop_lengths=[16, 8],
            win_lengths=[64, 32],
        ),
    ),
    use_mel_loss=False,
    use_dual_decoder=False,
)


def _optim_args(**overrides):
    """Return the arguments `build_optimizers` reads, plus overrides."""
    options = dict(
        optim="adam",
        optim_conf={"lr": 2.0e-4},
        optim2="adamw",
        optim2_conf={"lr": 1.0e-4},
        sharded_ddp=False,
    )
    options.update(overrides)
    return argparse.Namespace(**options)


@pytest.fixture(scope="module")
def model():
    """Build the reduced EnCodec model once for the module."""
    args = argparse.Namespace(
        codec="encodec", codec_conf=TINY_ENCODEC_CONF, model_conf={}
    )
    return GANCodecTask.build_model(args)


def test_add_arguments():
    """Ensure parser construction succeeds."""
    GANCodecTask.get_parser()


def test_add_arguments_help():
    """Ensure parser help exits cleanly."""
    parser = GANCodecTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Ensure main help exits cleanly."""
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=["--help"])


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Ensure main without args exits with usage."""
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """Ensure printed config can be parsed back."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        GANCodecTask.print_config(f)
    parser = GANCodecTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_data_names_match_espnet2(inference):
    """The copy must keep espnet2's required/optional data field names."""
    assert GANCodecTask.required_data_names(
        True, inference
    ) == ESPnet2GANCodecTask.required_data_names(True, inference)
    assert GANCodecTask.optional_data_names(
        True, inference
    ) == ESPnet2GANCodecTask.optional_data_names(True, inference)
    assert GANCodecTask.required_data_names(True, inference) == ("audio",)


def test_task_settings_match_espnet2():
    """The copy keeps espnet2's codec choices, trainer and optimizer count."""
    espnet2_choices = ESPnet2GANCodecTask.class_choices_list[0]
    assert codec_choices.name == espnet2_choices.name
    assert codec_choices.classes == espnet2_choices.classes
    assert codec_choices.default == espnet2_choices.default
    assert GANCodecTask.trainer is ESPnet2GANCodecTask.trainer
    assert GANCodecTask.num_optimizers == ESPnet2GANCodecTask.num_optimizers == 2


@pytest.mark.parametrize("train", [True, False])
def test_collate_fn_pads_audio_with_zeros(train):
    """The collate function pads the shorter waveform with zeros."""
    collate_fn = GANCodecTask.build_collate_fn(argparse.Namespace(), train)
    assert isinstance(collate_fn, CommonCollateFn)

    long = np.ones(6, dtype=np.float32)
    short = np.ones(4, dtype=np.float32)
    uttids, batch = collate_fn([("a", {"audio": long}), ("b", {"audio": short})])

    assert uttids == ["a", "b"]
    assert batch["audio"].shape == (2, 6)
    assert batch["audio_lengths"].tolist() == [6, 4]
    assert batch["audio"][1, 4:].tolist() == [0.0, 0.0]


def test_preprocess_fn_is_none_without_preprocessor():
    """`use_preprocessor: false` builds no preprocessing function."""
    args = argparse.Namespace(use_preprocessor=False)
    assert GANCodecTask.build_preprocess_fn(args, train=True) is None


@pytest.mark.parametrize(
    "iterator_type, expected_min_sample_size",
    [("chunk", 8000), ("sequence", -1)],
)
def test_preprocess_fn_settings(iterator_type, expected_min_sample_size):
    """Only the chunk iterator pads short utterances up to the chunk length."""
    args = argparse.Namespace(
        use_preprocessor=True, iterator_type=iterator_type, chunk_length=8000
    )
    preprocess_fn = GANCodecTask.build_preprocess_fn(args, train=True)

    assert isinstance(preprocess_fn, CommonPreprocessor)
    assert preprocess_fn.train is True
    assert preprocess_fn.speech_name == "audio"
    assert preprocess_fn.tokenizer is None
    assert preprocess_fn.force_single_channel is True
    assert preprocess_fn.audio_pad_value == 0.0
    assert preprocess_fn.min_sample_size == expected_min_sample_size


def test_preprocess_fn_pads_short_audio_for_chunks():
    """A waveform shorter than the chunk is zero-padded past the chunk length."""
    args = argparse.Namespace(
        use_preprocessor=True, iterator_type="chunk", chunk_length=8
    )
    preprocess_fn = GANCodecTask.build_preprocess_fn(args, train=True)

    out = preprocess_fn("utt1", {"audio": np.ones(5, dtype=np.float32)})

    assert out["audio"].shape == (9,)
    assert out["audio"][5:].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_build_model_builds_the_chosen_codec(model):
    """`codec` selects the class and `codec_conf` reaches its constructor."""
    assert isinstance(model, ESPnetGANCodecModel)
    assert isinstance(model.codec, Encodec)
    assert model.codec.generator.target_bandwidths == [0.5, 1]


def test_build_optimizers_splits_generator_and_discriminator(model):
    """The first optimizer holds the generator, the second the discriminator."""
    optim_g, optim_d = GANCodecTask.build_optimizers(_optim_args(), model)

    assert isinstance(optim_g, torch.optim.Adam)
    assert isinstance(optim_d, torch.optim.AdamW)
    assert optim_g.defaults["lr"] == 2.0e-4
    assert optim_d.defaults["lr"] == 1.0e-4

    def held(optimizer):
        return {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert held(optim_g) == {id(p) for p in model.codec.generator.parameters()}
    assert held(optim_d) == {id(p) for p in model.codec.discriminator.parameters()}
    assert held(optim_g).isdisjoint(held(optim_d))


@pytest.mark.parametrize("name", ["optim", "optim2"])
def test_build_optimizers_rejects_unknown_optimizer(model, name):
    """An unknown optimizer name is reported together with the valid ones."""
    args = _optim_args(**{name: "no_such_optimizer"})
    with pytest.raises(ValueError, match="no_such_optimizer"):
        GANCodecTask.build_optimizers(args, model)


def test_build_optimizers_sharded_ddp_needs_fairscale(model, monkeypatch):
    """`sharded_ddp` without fairscale installed is a clear error."""
    monkeypatch.setitem(sys.modules, "fairscale", None)
    with pytest.raises(RuntimeError, match="fairscale"):
        GANCodecTask.build_optimizers(_optim_args(sharded_ddp=True), model)


def test_build_optimizers_sharded_ddp_wraps_both_optimizers(model, monkeypatch):
    """`sharded_ddp` wraps each optimizer class in fairscale's OSS."""
    calls = []

    def oss(params, optim, **optim_conf):
        calls.append((list(params), optim, optim_conf))
        return optim(calls[-1][0], **optim_conf)

    fairscale = types.ModuleType("fairscale")
    fairscale.optim = types.SimpleNamespace(oss=types.SimpleNamespace(OSS=oss))
    monkeypatch.setitem(sys.modules, "fairscale", fairscale)

    optimizers = GANCodecTask.build_optimizers(_optim_args(sharded_ddp=True), model)

    assert len(optimizers) == 2
    (params_g, class_g, conf_g), (params_d, class_d, conf_d) = calls
    assert class_g is torch.optim.Adam
    assert class_d is torch.optim.AdamW
    assert conf_g == {"lr": 2.0e-4}
    assert conf_d == {"lr": 1.0e-4}
    assert len(params_g) == len(list(model.codec.generator.parameters()))
    assert len(params_d) == len(list(model.codec.discriminator.parameters()))
