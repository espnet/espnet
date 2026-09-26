from argparse import Namespace

import numpy as np
import pytest
import torch

from espnet2.tasks.gan_tok import GANSpeechTokenizerTask


def test_add_arguments():
    GANSpeechTokenizerTask.get_parser()


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


class DummyReconstructionObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 2)
        self.discriminator = torch.nn.Linear(2, 1)


class DummyASRObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.asr_model = torch.nn.Linear(2, 2)


class DummyGANModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = torch.nn.Linear(2, 2)
        self.asr_objective = DummyASRObjective()
        self.reconstruction_objective = DummyReconstructionObjective()


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
        for parameter in model.reconstruction_objective.discriminator.parameters()
    }

    assert main_ids.isdisjoint(discriminator_ids)
    assert discriminator_ids == expected_discriminator_ids
    assert id(model.tokenizer.weight) in main_ids
    assert id(model.asr_objective.asr_model.weight) in main_ids
    assert id(model.reconstruction_objective.generator.weight) in main_ids
    assert main_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert discriminator_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
