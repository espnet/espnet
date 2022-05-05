import configargparse
import pytest
import torch

from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.collate_fn import CommonCollateFn


class TestModel(AbsESPnetModel):
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
        model = TestModel()
        return model

    @classmethod
    def build_optimizers(cls, args, model):
        optim = torch.optim.Adam(model.layer1.parameters())
        optim2 = torch.optim.Adam(model.layer2.parameters())
        optimizers = [optim, optim2]
        return optimizers


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
