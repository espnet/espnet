from argparse import Namespace
from test.espnet2.tok.test_hifigan_loss import (
    TINY_DISCRIMINATOR_PARAMS,
    TINY_GENERATOR_PARAMS,
)
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch

from espnet2.tasks.gan_tok import GANSpeechTokenizerTask
from espnet2.tok.espnet_model import ESPnetGANSpeechTokenizerModel
from espnet2.train.gan_trainer import GANTrainer
from espnet2.utils.get_default_kwargs import get_default_kwargs


def test_add_arguments():
    parser = GANSpeechTokenizerTask.get_parser()
    args = parser.parse_args(["--token_list", "tokens.txt"])

    assert GANSpeechTokenizerTask.trainer is GANTrainer
    assert args.generator_first is True


def test_build_model_requires_skipping_inactive_discriminator():
    parser = GANSpeechTokenizerTask.get_parser()
    args = parser.parse_args(
        [
            "--token_list",
            "tokens.txt",
            "--model_conf",
            "reconstruction_weight=0.0",
            "--skip_discriminator_prob",
            "0.0",
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "skip_discriminator_prob must be 1.0 when " "reconstruction_weight is 0.0"
        ),
    ):
        GANSpeechTokenizerTask.build_model(args)


def test_add_arguments_help():
    parser = GANSpeechTokenizerTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        GANSpeechTokenizerTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        GANSpeechTokenizerTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        GANSpeechTokenizerTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as file:
        GANSpeechTokenizerTask.print_config(file)
    parser = GANSpeechTokenizerTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [False, True])
def test_required_data_names(inference):
    expected = ("speech",) if inference else ("speech", "text", "spembs")
    assert GANSpeechTokenizerTask.required_data_names(inference=inference) == expected


def test_optional_data_names():
    assert GANSpeechTokenizerTask.optional_data_names() == ()


def test_build_preprocess_fn_disabled():
    assert (
        GANSpeechTokenizerTask.build_preprocess_fn(
            Namespace(use_preprocessor=False), train=True
        )
        is None
    )


def test_build_collate_fn_keeps_speaker_embeddings_non_sequence():
    collate_fn = GANSpeechTokenizerTask.build_collate_fn(Namespace(), train=True)
    uttids, batch = collate_fn(
        [
            (
                "utt1",
                {
                    "speech": np.ones(3, dtype=np.float32),
                    "text": np.array([1, 2], dtype=np.int64),
                    "spembs": np.ones(4, dtype=np.float32),
                },
            ),
            (
                "utt2",
                {
                    "speech": np.ones(2, dtype=np.float32),
                    "text": np.array([1], dtype=np.int64),
                    "spembs": np.zeros(4, dtype=np.float32),
                },
            ),
        ]
    )

    assert uttids == ["utt1", "utt2"]
    assert batch["spembs"].shape == (2, 4)
    assert "spembs_lengths" not in batch


class DummyReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 2)
        self.discriminator = torch.nn.Linear(2, 1)


class DummyASRLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.asr_model = torch.nn.Linear(2, 2)


class DummyGANModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = torch.nn.Linear(2, 2)
        self.asr_loss = DummyASRLoss()
        self.reconstruction_loss = DummyReconstructionLoss()


def test_build_optimizers_separates_discriminator_parameters():
    model = DummyGANModel()
    args = Namespace(
        optim="adam",
        optim2="adam",
        optim_conf={"lr": 1.0e-3},
        optim2_conf={"lr": 2.0e-3},
        sharded_ddp=False,
    )

    main_optimizer, discriminator_optimizer = GANSpeechTokenizerTask.build_optimizers(
        args, model
    )
    main_ids = {
        id(parameter)
        for group in main_optimizer.param_groups
        for parameter in group["params"]
    }
    discriminator_ids = {
        id(parameter)
        for group in discriminator_optimizer.param_groups
        for parameter in group["params"]
    }
    expected_discriminator_ids = {
        id(parameter)
        for parameter in model.reconstruction_loss.discriminator.parameters()
    }

    assert main_ids.isdisjoint(discriminator_ids)
    assert discriminator_ids == expected_discriminator_ids
    assert id(model.tokenizer.weight) in main_ids
    assert id(model.asr_loss.asr_model.weight) in main_ids
    assert id(model.reconstruction_loss.generator.weight) in main_ids
    assert main_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert discriminator_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)


def test_build_model_and_run_generator_discriminator_steps(tmp_path):
    token_list = tmp_path / "tokens.txt"
    token_list.write_text("<blank>\na\nb\n<sos/eos>\n", encoding="utf-8")
    centroid_path = tmp_path / "kmeans.mdl"
    joblib.dump(
        SimpleNamespace(cluster_centers_=np.zeros((4, 2), dtype=np.float32)),
        centroid_path,
    )

    parser = GANSpeechTokenizerTask.get_parser()
    args = parser.parse_args(["--token_list", str(token_list)])
    args.frontend_conf = {
        "n_fft": 8,
        "win_length": 8,
        "hop_length": 2,
        "n_mels": 2,
        "frontend_conf": None,
    }
    args.quantizer_conf = {"centroid_path": str(centroid_path)}
    args.asr_embedding_conf = {
        "embed_dim": 4,
        "use_positional_encoding": False,
    }
    args.asr_encoder_conf = {
        "num_layers": 1,
        "hidden_size": 4,
        "output_size": 4,
        "subsample": [1],
    }
    args.asr_model_conf.update(
        ctc_weight=1.0,
        report_cer=False,
        report_wer=False,
    )
    args.reconstruction_loss_conf = {
        "token_embed_dim": 3,
        "speaker_embed_dim": 2,
        "segment_size": 4,
        "generator_params": TINY_GENERATOR_PARAMS,
        "discriminator_params": TINY_DISCRIMINATOR_PARAMS,
        "use_feat_match_loss": False,
        "use_mel_loss": False,
    }
    args.model_conf = {"extract_feats_in_collect_stats": True}

    model = GANSpeechTokenizerTask.build_model(args)
    assert model.extract_feats_in_collect_stats is True
    assert (
        model.loss_weights["reconstruction"]
        == get_default_kwargs(ESPnetGANSpeechTokenizerModel)["reconstruction_weight"]
    )
    optimizers = GANSpeechTokenizerTask.build_optimizers(args, model)
    batch = {
        "speech": torch.randn(1, 32),
        "speech_lengths": torch.tensor([32]),
        "text": torch.tensor([[1, 2]], dtype=torch.long),
        "text_lengths": torch.tensor([2]),
        "spembs": torch.randn(1, 2),
    }

    assert model.tokenizer.current_step.item() == 0
    assert model.tokenizer.quantizer.num_updates.item() == 0
    for optim_idx, (forward_generator, optimizer) in enumerate(
        zip((True, False), optimizers)
    ):
        optimizer.zero_grad()
        result = model(forward_generator=forward_generator, **batch)
        assert result["optim_idx"] == optim_idx
        result["loss"].backward()
        optimizer.step()
        assert model.tokenizer.current_step.item() == 1
        assert model.tokenizer.quantizer.num_updates.item() == 1
        assert (
            model.reconstruction_loss.has_cached_generator_outputs is forward_generator
        )
