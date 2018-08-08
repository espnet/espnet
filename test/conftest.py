import pytest
import torch

from e2e_asr_th import torch_is_old


@pytest.fixture()
def torch_fixture():
    # TODO(karita): this is workaround before going 0.4.0
    # we have to rewrite with torch.no_grad instead of torch.set_grad_enabled
    if not torch_is_old:
        torch.set_grad_enabled(True)
