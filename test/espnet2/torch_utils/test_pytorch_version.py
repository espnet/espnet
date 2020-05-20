from espnet2.torch_utils.pytorch_version import pytorch_cudnn_version


def test_pytorch_cudnn_version():
    print(pytorch_cudnn_version())
