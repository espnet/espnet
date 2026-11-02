import argparse
import logging

import configargparse
import pytest
import torch
import yaml

from espnet2.tasks import abs_task
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.collate_fn import CommonCollateFn


class DummyModel(AbsESPnetModel):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, 1)
        self.layer2 = torch.nn.Linear(1, 1)

    def collect_feats(self):
        return {}

    def forward(self, x, x_lengths, **kwargs):
        x = self.layer1(x)
        x = self.layer2(x)
        retval = {
            "loss": x.mean(),
            "stats": {"loss": x.mean()},
            "weight": len(x),
            "optim_idx": torch.randint(0, 2, [1]),
        }
        return force_gatherable(retval, device=x.device)


class TestTask(AbsTask):
    num_optimizers: int = 2

    @classmethod
    def add_task_arguments(cls, parser):
        pass

    @classmethod
    def build_collate_fn(cls, args, train):
        return CommonCollateFn()

    @classmethod
    def build_preprocess_fn(cls, args, train):
        return None

    @classmethod
    def required_data_names(cls, train=True, inference=False):
        if not inference:
            retval = ("x",)
        else:
            # Recognition mode
            retval = ("x",)
        return retval

    @classmethod
    def optional_data_names(cls, train=True, inference=False):
        retval = ()
        return retval

    @classmethod
    def build_model(cls, args):
        model = DummyModel()
        return model

    @classmethod
    def build_optimizers(cls, args, model):
        optim = torch.optim.Adam(model.layer1.parameters())
        optim2 = torch.optim.Adam(model.layer2.parameters())
        optimizers = [optim, optim2]
        return optimizers


class InitTask(TestTask):
    """A task that initializes what it builds, as the real ones do."""

    @classmethod
    def build_model(cls, args):
        model = DummyModel()
        if getattr(args, "init", None) is not None:
            initialize(model, args.init)
        return model


@pytest.mark.parametrize("parser", [configargparse.ArgumentParser(), None])
def test_add_arguments(parser):
    AbsTask.get_parser()


def test_add_arguments_help():
    parser = AbsTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        AbsTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        AbsTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        AbsTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        AbsTask.print_config(f)
    parser = AbsTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


def test_build_model_from_file_accepts_lightning_checkpoint(tmp_path):
    config_file = tmp_path / "config.yaml"
    model_file = tmp_path / "model.pth"

    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump({}, f)

    expected = DummyModel()
    with torch.no_grad():
        expected.layer1.weight.fill_(1.25)
        expected.layer1.bias.fill_(2.5)
        expected.layer2.weight.fill_(3.75)
        expected.layer2.bias.fill_(5.0)

    torch.save({"state_dict": expected.state_dict()}, model_file)

    loaded_model, args = TestTask.build_model_from_file(
        config_file=config_file,
        model_file=model_file,
        device="cpu",
    )

    assert isinstance(args, argparse.Namespace)
    for name, parameter in expected.state_dict().items():
        assert torch.equal(loaded_model.state_dict()[name], parameter)


# FIXME(kamo): This is an integration test, so it's hard to reduce time
@pytest.mark.execution_timeout(50)
def test_main(tmp_path):
    train_text = tmp_path / "train.txt"
    with train_text.open("w") as f:
        f.write("a 10,1\n")

    TestTask.main(
        cmd=[
            "--output_dir",
            str(tmp_path / "out"),
            "--train_data_path_and_name_and_type",
            f"{train_text},x,rand_float",
            "--train_shape_file",
            str(train_text),
            "--valid_data_path_and_name_and_type",
            f"{train_text},x,rand_float",
            "--valid_shape_file",
            str(train_text),
            "--batch_size",
            "1",
            "--batch_type",
            "unsorted",
            "--max_epoch",
            "1",
        ]
    )


def test_an_init_this_version_removed_does_not_stop_a_model_loading(tmp_path, caplog):
    """A checkpoint says how it was initialised; this version may not know it.

    `init: chainer` was one of six choices until it was removed in June 2025,
    and espnet_model_zoo's daily run found it still published: jv_openslr35,
    trained on espnet 0.9.7, failed with `Unknown initialization: chainer`.
    The setting cannot matter when loading - the weights come from the file -
    so it is ignored, with a line saying so.
    """
    config_file = tmp_path / "config.yaml"
    model_file = tmp_path / "model.pth"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"init": "chainer"}, f)
    torch.save(DummyModel().state_dict(), model_file)

    with caplog.at_level(logging.WARNING):
        _, args = TestTask.build_model_from_file(
            config_file=config_file, model_file=model_file, device="cpu"
        )

    assert args.init is None
    assert "init=chainer" in caplog.text


def test_without_a_checkpoint_an_unknown_init_is_still_an_error(tmp_path):
    """Nothing overwrites the parameters then, so the setting is all there is.

    Building from a config alone - no model file - leaves the model with
    whatever the initialization gives it. Quietly using the default instead
    of the one the config asked for would be a different model with no sign
    of it.
    """
    config_file = tmp_path / "config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"init": "chainer"}, f)

    with pytest.raises(ValueError, match="Unknown initialization: chainer"):
        InitTask.build_model_from_file(config_file=config_file, device="cpu")


def test_an_init_this_version_has_is_left_alone(tmp_path):
    """Not a blanket "ignore init when loading": it still applies.

    load_state_dict here is not strict, so a parameter the checkpoint does
    not carry keeps whatever the initialization gave it.
    """
    args = argparse.Namespace(init="xavier_uniform")

    abs_task._drop_init_this_version_removed(args)

    assert args.init == "xavier_uniform"


def test_a_misspelled_init_is_still_reported_when_training():
    """Ignoring is for loading a checkpoint; building one still refuses."""
    with pytest.raises(ValueError, match="Unknown initialization: xavier"):
        initialize(DummyModel(), "xavier")
