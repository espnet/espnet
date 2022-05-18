from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complexnn import (
    ComplexBatchNorm,
    ComplexConv2d,
    ComplexConvTranspose2d,
    NavieComplexLSTM,
    complex_cat,
)
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
EPS = torch.finfo(torch.double).eps


class DCCRNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        rnn_layer: int = 2,
        rnn_units: int = 256,
        masking_mode: str = "E",
        use_clstm: bool = True,
        bidirectional: bool = False,
        use_cbn: bool = False,
        kernel_size: int = 5,
        kernel_num: List[int] = [32, 64, 128, 256, 256, 256],
        use_builtin_complex: bool = True,
        use_noise_mask: bool = False,
    ):
        """DCCRN separator.

        Args:
            input_dim (int): input dimension。
            num_spk (int, optional): number of speakers. Defaults to 1.
            rnn_layer (int, optional): number of lstm layers in the crn. Defaults to 2.
            rnn_units (int, optional): rnn units. Defaults to 128.
            masking_mode (str, optional): usage of the estimated mask. Defaults to "E".
            use_clstm (bool, optional): whether use complex LSTM. Defaults to False.
            bidirectional (bool, optional): whether use BLSTM. Defaults to False.
            use_cbn (bool, optional): whether use complex BN. Defaults to False.
            kernel_size (int, optional): convolution kernel size. Defaults to 5.
            kernel_num (list, optional): output dimension of each layer of the encoder.
            use_builtin_complex (bool, optional): torch.complex if True,
                                                else ComplexTensor.
            use_noise_mask (bool, optional): whether to estimate the mask of noise.
        """
        super().__init__()
        self.use_builtin_complex = use_builtin_complex
        self._num_spk = num_spk
        self.use_noise_mask = use_noise_mask
        if masking_mode not in ["C", "E", "R"]:
            raise ValueError("Unsupported masking mode: %s" % masking_mode)
        # Network config
        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layer
        self.kernel_size = kernel_size
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        fac = 2 if bidirectional else 1

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1])
                    if not use_cbn
                    else ComplexBatchNorm(self.kernel_num[idx + 1]),
                    nn.PReLU(),
                )
            )
        hidden_dim = (input_dim - 1 + 2 ** (len(self.kernel_num) - 1) - 1) // (
            2 ** (len(self.kernel_num) - 1)
        )
        hidden_dim = hidden_dim if hidden_dim > 0 else 1

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layer):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1]
                        if idx == 0
                        else self.rnn_units * fac,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1]
                        if idx == rnn_layer - 1
                        else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False,
            )
            self.tranform = nn.Linear(
                self.rnn_units * fac, hidden_dim * self.kernel_num[-1]
            )

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1])
                        if not use_cbn
                        else ComplexBatchNorm(self.kernel_num[idx - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1] * (self._num_spk + 1)
                            if self.use_noise_mask
                            else self.kernel_num[idx - 1] * self._num_spk,
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        ),
                    )
                )

        self.flatten_parameters()

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # shape (B, T, F) --> (B, F, T)
        specs = input.permute(0, 2, 1)
        real, imag = specs.real, specs.imag

        # # shape (B, F, T)
        # spec_mags = torch.sqrt(real**2 + imag**2 + 1e-8)
        # # shape (B, F, T)
        # spec_phase = torch.atan2(imag, real)
        # shape (B, 2, F, T)
        cspecs = torch.stack([real, imag], 1)
        # shape (B, 2, F-1, T)
        cspecs = cspecs[:, :, 1:]

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)
        # shape (B, C, F, T)
        batch_size, channels, dims, lengths = out.size()
        # shape (T, B, C, F)
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            # shape (T, B, C // 2, F)
            r_rnn_in = out[:, :, : channels // 2]
            # shape (T, B, C // 2, F)
            i_rnn_in = out[:, :, channels // 2 :]
            # shape (T, B, C // 2 * F)
            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels // 2 * dims]
            )
            # shape (T, B, C // 2 * F)
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels // 2 * dims]
            )
            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])
            # shape (T, B, C // 2, F)
            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels // 2, dims]
            )
            # shape (T, B, C // 2, F)
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels // 2, dims]
            )
            # shape (T, B, C, F)
            out = torch.cat([r_rnn_in, i_rnn_in], 2)

        else:
            # shape (T, B, C*F)
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            # shape (T, B, C, F)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])
        # shape (B, C, F, T)
        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            # skip connection
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        # out shape = (B, 2*num_spk, F-1, T) if self.use_noise_mask == False
        # else (B, 2*(num_spk+1), F-1, T)

        masks = self.create_masks(out)
        masked = self.apply_masks(masks, real, imag)
        others = OrderedDict(
            zip(
                ["mask_spk{}".format(i + 1) for i in range(self.num_spk)],
                masks,
            )
        )

        if self.use_noise_mask:
            others["mask_noise1"] = masks[-1]
            others["noise1"] = masked.pop(-1)

        return (masked, ilens, others)

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def create_masks(self, mask_tensor: torch.Tensor):
        """create estimated mask for each speaker

        Args:
            mask_tensor (torch.Tensor): output of decoder, shape(B, 2*num_spk, F-1, T)
        """
        if self.use_noise_mask:
            assert mask_tensor.shape[1] == 2 * (self._num_spk + 1), mask_tensor.shape[1]
        else:
            assert mask_tensor.shape[1] == 2 * self._num_spk, mask_tensor.shape[1]

        masks = []
        for idx in range(mask_tensor.shape[1] // 2):
            # shape (B, F-1, T)
            mask_real = mask_tensor[:, idx * 2]
            # shape (B, F-1, T)
            mask_imag = mask_tensor[:, idx * 2 + 1]
            # shape (B, F, T)
            mask_real = F.pad(mask_real, [0, 0, 1, 0])
            # shape (B, F, T)
            mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

            # mask shape (B, T, F)
            if is_torch_1_9_plus and self.use_builtin_complex:
                complex_mask = torch.complex(
                    mask_real.permute(0, 2, 1), mask_imag.permute(0, 2, 1)
                )
            else:
                complex_mask = ComplexTensor(
                    mask_real.permute(0, 2, 1), mask_imag.permute(0, 2, 1)
                )

            masks.append(complex_mask)

        return masks

    def apply_masks(
        self,
        masks: List[Union[torch.Tensor, ComplexTensor]],
        real: torch.Tensor,
        imag: torch.Tensor,
    ):
        """apply masks

        Args:
            masks : est_masks, [(B, T, F), ...]
            real (torch.Tensor): real part of the noisy spectrum, (B, F, T)
            imag (torch.Tensor): imag part of the noisy spectrum, (B, F, T)

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, F), ...]
        """
        masked = []
        for i in range(len(masks)):
            # shape (B, T, F) --> (B, F, T)
            mask_real = masks[i].real.permute(0, 2, 1)
            mask_imag = masks[i].imag.permute(0, 2, 1)
            if self.masking_mode == "E":
                # shape (B, F, T)
                spec_mags = torch.sqrt(real**2 + imag**2 + 1e-8)
                # shape (B, F, T)
                spec_phase = torch.atan2(imag, real)
                mask_mags = (mask_real**2 + mask_imag**2) ** 0.5
                # mask_mags = (mask_real ** 2 + mask_imag ** 2 + EPS) ** 0.5
                real_phase = mask_real / (mask_mags + EPS)
                imag_phase = mask_imag / (mask_mags + EPS)
                # mask_phase = torch.atan2(imag_phase + EPS, real_phase + EPS)
                mask_phase = torch.atan2(imag_phase, real_phase)
                mask_mags = torch.tanh(mask_mags)
                est_mags = mask_mags * spec_mags
                est_phase = spec_phase + mask_phase
                real = est_mags * torch.cos(est_phase)
                imag = est_mags * torch.sin(est_phase)
            elif self.masking_mode == "C":
                real, imag = (
                    real * mask_real - imag * mask_imag,
                    real * mask_imag + imag * mask_real,
                )
            elif self.masking_mode == "R":
                real, imag = real * mask_real, imag * mask_imag

            # shape (B, F, T) --> (B, T, F)
            if is_torch_1_9_plus and self.use_builtin_complex:
                masked.append(
                    torch.complex(real.permute(0, 2, 1), imag.permute(0, 2, 1))
                )
            else:
                masked.append(
                    ComplexTensor(real.permute(0, 2, 1), imag.permute(0, 2, 1))
                )
        return masked

    @property
    def num_spk(self):
        return self._num_spk
