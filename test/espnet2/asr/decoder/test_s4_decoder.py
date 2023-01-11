import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet.nets.batch_beam_search import BatchBeamSearch

# Check to have torch.linalg
is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")


@pytest.mark.parametrize("input_layer", ["embed"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("n_layers", [3, 6])
@pytest.mark.parametrize("residual", ["residual", None])
@pytest.mark.parametrize("norm", ["layer", "batch"])
@pytest.mark.parametrize("drop_path", [0.0, 0.1])
def test_S4Decoder_backward(input_layer, prenorm, n_layers, norm, residual, drop_path):
    # Skip test for the lower pytorch versions
    if not is_torch_1_7_plus:
        return
    layer = [
        {"_name_": "s4", "keops": True},  # Do not use custom Cauchy kernel (CUDA)
        {"_name_": "mha", "n_head": 4},
        {"_name_": "ff"},
    ]
    decoder = S4Decoder(
        vocab_size=10,
        encoder_output_size=12,
        input_layer=input_layer,
        prenorm=prenorm,
        n_layers=n_layers,
        layer=layer,
        norm=norm,
        residual=residual,
        drop_path=drop_path,
    )
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


@pytest.mark.parametrize("input_layer", ["embed"])
@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("n_layers", [3, 6])
@pytest.mark.parametrize("residual", ["residual", None])
@pytest.mark.parametrize("norm", ["layer", "batch"])
@pytest.mark.parametrize("drop_path", [0.0, 0.1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_S4Decoder_batch_beam_search(
    input_layer, prenorm, n_layers, norm, residual, drop_path, dtype
):
    # Skip test for the lower pytorch versions
    if not is_torch_1_7_plus:
        return
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)
    encoder_output_size = 4
    layer = [
        {"_name_": "s4", "keops": True},  # Do not use custom Cauchy kernel
        {"_name_": "mha", "n_head": 4},
        {"_name_": "ff"},
    ]

    decoder = S4Decoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        input_layer=input_layer,
        prenorm=prenorm,
        n_layers=n_layers,
        layer=layer,
        norm=norm,
        residual=residual,
        drop_path=drop_path,
    )
    beam = BatchBeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": decoder},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype).eval()

    for module in beam.nn_dict.test.modules():
        if hasattr(module, "setup_step"):
            module.setup_step()

    enc = torch.randn(10, encoder_output_size).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=0.0,
            minlenratio=0.0,
        )
