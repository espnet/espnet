import torch
import torch.nn as nn

class Houlsby_Adapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        bottleneck: int,
    ):
        super(Houlsby_Adapter, self).__init__()

        self.houlsby_adapter = nn.Sequential(
            nn.Linear(input_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, input_size),
        )

    def forward(self, x):
        return self.houlsby_adapter(x)



