from distutils.version import LooseVersion
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet2.enh.layers.complex_utils import is_complex


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class MaskEstimator(torch.nn.Module):
    def __init__(
        self, type, idim, layers, units, projs, dropout, nmask=1, nonlinear="sigmoid"
    ):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.nmask = nmask
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(projs, idim) for _ in range(nmask)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh", "crelu"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = nonlinear

    def forward(
        self, xs: Union[torch.Tensor, ComplexTensor], ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """Mask estimator forward function.

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        _, _, C, input_length = xs.size()
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        if is_complex(xs):
            xs = (xs.real**2 + xs.imag**2) ** 0.5
        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        xs = xs.contiguous().view(-1, xs.size(-2), xs.size(-1))
        # ilens: (B,) -> ilens_: (B * C)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        # xs: (B * C, T, F) -> xs: (B * C, T, D)
        xs, _, _ = self.brnn(xs, ilens_)
        # xs: (B * C, T, D) -> xs: (B, C, T, D)
        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []
        for linear in self.linears:
            # xs: (B, C, T, D) -> mask:(B, C, T, F)
            mask = linear(xs)

            if self.nonlinear == "sigmoid":
                mask = torch.sigmoid(mask)
            elif self.nonlinear == "relu":
                mask = torch.relu(mask)
            elif self.nonlinear == "tanh":
                mask = torch.tanh(mask)
            elif self.nonlinear == "crelu":
                mask = torch.clamp(mask, min=0, max=1)
            # Zero padding
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            # (B, C, T, F) -> (B, F, C, T)
            mask = mask.permute(0, 3, 1, 2)

            # Take cares of multi gpu cases: If input_length > max(ilens)
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens
