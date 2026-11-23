"""Tests for ESPnet3 enhancement task wiring."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.train.preprocessor import EnhPreprocessor
from espnet3.systems.esp2_enh.task import EnhancementTask


def test_add_arguments():
    EnhancementTask.get_parser()


def test_add_arguments_help():
    parser = EnhancementTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        EnhancementTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        EnhancementTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        EnhancementTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        EnhancementTask.print_config(f)
    parser = EnhancementTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


def test_data_names_switch_between_training_and_inference():
    assert EnhancementTask.required_data_names(train=True) == ("speech_ref1",)
    assert EnhancementTask.required_data_names(inference=True) == ("speech_mix",)

    optional = EnhancementTask.optional_data_names()
    assert optional[0] == "speech_mix"
    assert "speech_ref1" not in optional
    for name in ("speech_ref2", "dereverb_ref1", "noise_ref1", "category", "fs"):
        assert name in optional


def _model_args(**overrides):
    args = dict(
        encoder="stft",
        encoder_conf={"n_fft": 32, "hop_length": 8},
        separator="rnn",
        separator_conf={"num_spk": 1, "layer": 1, "unit": 8},
        decoder="stft",
        decoder_conf={"n_fft": 32, "hop_length": 8},
        criterions=[
            {"name": "si_snr", "conf": {}, "wrapper": "fixed_order", "wrapper_conf": {}}
        ],
        model_conf={},
        init=None,
    )
    args.update(overrides)
    return Namespace(**args)


def test_build_model_wires_parts_and_runs_forward():
    model = EnhancementTask.build_model(_model_args())

    assert isinstance(model, ESPnetEnhancementModel)
    assert len(model.loss_wrappers) == 1
    assert isinstance(model.loss_wrappers[0], FixedOrderSolver)

    speech = torch.randn(2, 400)
    lengths = torch.tensor([400, 400])
    loss, stats, _ = model(
        speech_mix=speech, speech_mix_lengths=lengths, speech_ref1=speech
    )
    assert torch.isfinite(loss)
    assert "si_snr_loss" in stats


def test_build_model_without_criterions_has_no_loss_wrappers():
    model = EnhancementTask.build_model(_model_args(criterions=None))

    assert model.loss_wrappers == []


def test_build_model_applies_init():
    torch.manual_seed(0)
    model = EnhancementTask.build_model(_model_args(init="xavier_uniform"))

    # xavier_uniform zeroes biases.
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    assert biases and all(torch.count_nonzero(b) == 0 for b in biases)


def test_build_preprocess_fn_without_preprocessor_returns_none():
    assert (
        EnhancementTask.build_preprocess_fn(Namespace(preprocessor=None), True) is None
    )


def test_build_preprocess_fn_enh_applies_conf_over_defaults():
    args = Namespace(preprocessor="enh", preprocessor_conf={"sample_rate": 16000})

    preprocess = EnhancementTask.build_preprocess_fn(args, True)

    assert isinstance(preprocess, EnhPreprocessor)
    assert preprocess.train is True
    assert preprocess.sample_rate == 16000


def test_build_preprocess_fn_rejects_unknown_preprocessor():
    args = Namespace(preprocessor="default", preprocessor_conf={})
    with pytest.raises(ValueError, match="not supported"):
        EnhancementTask.build_preprocess_fn(args, True)


def test_collate_fn_pads_with_zeros():
    collate = EnhancementTask.build_collate_fn(Namespace(), True)

    ids, batch = collate(
        [
            ("a", {"speech_mix": np.ones(3, dtype=np.float32)}),
            ("b", {"speech_mix": np.ones(5, dtype=np.float32)}),
        ]
    )

    assert ids == ["a", "b"]
    assert batch["speech_mix"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["speech_mix_lengths"].tolist() == [3, 5]
