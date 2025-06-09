import pytest
import torch

from espnet.nets.asr_interface import dynamic_import_asr


@pytest.mark.parametrize(
    "dtype, device, model, conf",
    [
        (dtype, device, nn, conf)
        for nn, conf in [
            (
                "transformer",
                dict(adim=4, eunits=3, dunits=3, elayers=2, dlayers=2, mtlalpha=0.0),
            ),
            (
                "transformer",
                dict(
                    adim=4,
                    eunits=3,
                    dunits=3,
                    elayers=2,
                    dlayers=2,
                    mtlalpha=0.5,
                    ctc_type="builtin",
                ),
            ),
            (
                "transformer",
                dict(
                    adim=4,
                    eunits=3,
                    dunits=3,
                    elayers=2,
                    dlayers=2,
                    mtlalpha=0.5,
                    ctc_type="builtin",
                ),
            ),
            (
                "rnn",
                dict(adim=4, eunits=3, dunits=3, elayers=2, dlayers=2, mtlalpha=0.0),
            ),
            (
                "rnn",
                dict(
                    adim=4,
                    eunits=3,
                    dunits=3,
                    elayers=2,
                    dlayers=2,
                    mtlalpha=0.5,
                    ctc_type="builtin",
                ),
            ),
            (
                "rnn",
                dict(
                    adim=4,
                    eunits=3,
                    dunits=3,
                    elayers=2,
                    dlayers=2,
                    mtlalpha=0.5,
                    ctc_type="builtin",
                ),
            ),
        ]
        for dtype in ("float16", "float32", "float64")
        for device in ("cpu", "cuda")
    ],
)
def test_train_pytorch_dtype(dtype, device, model, conf):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")

    idim = 10
    odim = 10
    model = dynamic_import_asr(model, "pytorch").build(idim, odim, **conf)
    dtype = getattr(torch, dtype)
    device = torch.device(device)
    model.to(dtype=dtype, device=device)

    x = torch.rand(2, 10, idim, dtype=dtype, device=device)
    ilens = torch.tensor([10, 7], device=device)
    y = torch.randint(1, odim, (2, 3), device=device)
    opt = torch.optim.Adam(model.parameters())
    loss = model(x, ilens, y)
    assert loss.dtype == dtype
    model.zero_grad()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    opt.step()
