import torch
import torch.nn as nn
import espnet2.gan_svs.pits.modules as modules


class YingDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        yin_start,
        yin_scope,
        yin_shift_range,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = yin_scope
        self.out_channels = yin_scope
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.yin_start = yin_start
        self.yin_scope = yin_scope
        self.yin_shift_range = yin_shift_range

        self.pre = nn.Conv1d(self.in_channels, hidden_channels, 1)
        self.dec = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, self.out_channels, 1)

    def crop_scope(
        self, x, yin_start, scope_shift
    ):  # x: tensor [B,C,T] #scope_shift: tensor [B]
        return torch.stack(
            [
                x[
                    i,
                    yin_start
                    + scope_shift[i] : yin_start
                    + self.yin_scope
                    + scope_shift[i],
                    :,
                ]
                for i in range(x.shape[0])
            ],
            dim=0,
        )

    def infer(self, z_yin, z_mask, g=None):
        B = z_yin.shape[0]
        scope_shift = torch.randint(
            -self.yin_shift_range, self.yin_shift_range, (B,), dtype=torch.int
        )
        z_yin_crop = self.crop_scope(z_yin, self.yin_start, scope_shift)
        x = self.pre(z_yin_crop) * z_mask
        x = self.dec(x, z_mask, g=g)
        yin_hat_crop = self.proj(x) * z_mask
        return yin_hat_crop

    def forward(self, z_yin, yin_gt, z_mask, g=None):
        B = z_yin.shape[0]
        scope_shift = torch.randint(
            -self.yin_shift_range, self.yin_shift_range, (B,), dtype=torch.int
        )
        z_yin_crop = self.crop_scope(z_yin, self.yin_start, scope_shift)
        yin_gt_shifted_crop = self.crop_scope(yin_gt, self.yin_start, scope_shift)
        yin_gt_crop = self.crop_scope(
            yin_gt, self.yin_start, torch.zeros_like(scope_shift)
        )
        x = self.pre(z_yin_crop) * z_mask
        x = self.dec(x, z_mask, g=g)
        yin_hat_crop = self.proj(x) * z_mask
        return yin_gt_crop, yin_gt_shifted_crop, yin_hat_crop, z_yin_crop, scope_shift
