import numpy as np
import torch

from espnet2.torch_utils.load_pretrained_model import load_pretrained_model


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, 1)
        self.layer2 = torch.nn.Linear(2, 2)


def test_load_pretrained_model_all(tmp_path):
    model_src = Model()
    torch.save(model_src.state_dict(), tmp_path / "model.pth")

    model_dst = Model()
    load_pretrained_model(f"{tmp_path}/model.pth", model_dst, "cpu")

    for k in model_dst.state_dict():
        np.testing.assert_array_equal(
            model_dst.state_dict()[k].numpy(), model_src.state_dict()[k].numpy()
        )


def test_load_pretrained_model_layer1_layer1(tmp_path):
    model_src = Model()
    torch.save(model_src.state_dict(), tmp_path / "model.pth")

    model_dst = Model()
    load_pretrained_model(f"{tmp_path}/model.pth:layer1:layer1", model_dst, "cpu")

    for k in model_dst.state_dict():
        if k.startswith("layer1"):
            np.testing.assert_array_equal(
                model_dst.state_dict()[k].numpy(), model_src.state_dict()[k].numpy()
            )


def test_load_pretrained_model_exclude(tmp_path):
    model_src = Model()
    torch.save(model_src.state_dict(), tmp_path / "model.pth")

    model_dst = Model()
    load_pretrained_model(f"{tmp_path}/model.pth:::layer2", model_dst, "cpu")

    for k in model_dst.state_dict():
        if not k.startswith("layer2"):
            np.testing.assert_array_equal(
                model_dst.state_dict()[k].numpy(), model_src.state_dict()[k].numpy()
            )


def test_load_pretrained_model_layer1(tmp_path):
    model_src = Model()
    torch.save(model_src.layer1.state_dict(), tmp_path / "layer1.pth")

    model_dst = Model()
    load_pretrained_model(f"{tmp_path}/layer1.pth::layer1", model_dst, "cpu")

    for k in model_dst.state_dict():
        if k.startswith("layer1"):
            np.testing.assert_array_equal(
                model_dst.state_dict()[k].numpy(), model_src.state_dict()[k].numpy()
            )
