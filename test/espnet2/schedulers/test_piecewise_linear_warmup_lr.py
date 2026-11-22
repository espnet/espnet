import io

import torch

from espnet2.schedulers.piecewise_linear_warmup_lr import PiecewiseLinearWarmupLR


def test_PiecewiseLinearWarumupLR():
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), 0.1)
    sch = PiecewiseLinearWarmupLR(opt)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    assert lr != lr2


def test_piecewise_linear_warmup_lr_checkpoint_is_weights_only_loadable():
    """np.interp returns numpy.float64; torch>=2.6 refuses it on resume."""
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(linear.parameters(), lr=1e-3)
    sch = PiecewiseLinearWarmupLR(
        opt, warmup_steps_list=[0, 10], warmup_lr_list=[0.0, 1e-3]
    )
    for _ in range(3):
        opt.step()
        sch.step()

    assert isinstance(opt.param_groups[0]["lr"], float)
    assert isinstance(sch._last_lr[0], float)

    buf = io.BytesIO()
    torch.save({"opt": opt.state_dict(), "sch": sch.state_dict()}, buf)
    buf.seek(0)
    torch.load(buf, weights_only=True)  # the actual resume path
