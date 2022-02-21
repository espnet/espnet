from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.enh.layers.complexnn import (
    ComplexBatchNorm,
    ComplexConv2d,
    ComplexConvTranspose2d,
    NavieComplexLSTM,
    complex_cat,
)
from espnet2.enh.separator.abs_separator import AbsSeparator
from torch_complex.tensor import ComplexTensor

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")
EPS = torch.finfo(torch.double).eps


class DCCRNSeparator(AbsSeparator):
    def __init__(
        self,
        idim,
        rnn_layer=2,
        rnn_units=256,
        fft_len=512,
        masking_mode="E",
        use_clstm=True,
        bidirectional=False,
        use_cbn=False,
        kernel_size=5,
        kernel_num=[32, 64, 128, 256, 256, 256],
        use_builtin_complex: bool = True,
    ):
        """DCCRN separator

        Args:
            num_spk (int, optional): number of speakers. Defaults to 1.
            rnn_layer (int, optional): number of lstm layers in the crn. Defaults to 2.
            rnn_units (int, optional): number of features in the hidden state,
                                       for complex-lstm, rnn_units = real+imag. Defaults to 128.
            fft_len (int, optional): n_fft. Defaults to 512.
            masking_mode (str, optional): decide how to use the estimated mask. Defaults to "E".
            use_clstm (bool, optional): whether to use complex LSTM or not. Defaults to False.
            bidirectional (bool, optional): whether to use bidirectional LSTM or not. Defaults to False.
            use_cbn (bool, optional): whether to use complex batch normalization. Defaults to False.
            kernel_size (int, optional): convolution kernel size. Defaults to 5.
            kernel_num (list, optional): output dimension of each convolution layer of the encoder(decoder).
                                         Defaults to [16, 32, 64, 128, 256, 256].
            use_builtin_complex (bool, optional): use torch.complex if True, else use ComplexTensor.

        References
        - [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
                Yanxin Hu et al. https://arxiv.org/abs/2008.00264
        - [2] : https://github.com/huyanxin/DeepComplexCRN
        """
        super().__init__()
        self.use_builtin_complex = use_builtin_complex

        # Network config
        self.rnn_units = rnn_units
        self.fft_len = fft_len
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
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layer):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1]
                        if idx == 0
                        else self.rnn_units,
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
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        ),
                    )
                )

        self.flatten_parameters()

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, F]
            ilens (torch.Tensor): input lengths [Batch]

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
        # shape (B, F, T)
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        # shape (B, F, T)
        spec_phase = torch.atan2(imag, real)
        # shape (B, 2*F, T)
        cspecs = torch.stack([real, imag], 1)
        # shape (B, 2*F, T-1)
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
        # out shape (B, 2, F, T)

        # shape (B, F-1, T)
        mask_real = out[:, 0]
        # shape (B, F-1, T)
        mask_imag = out[:, 1]
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

        others = OrderedDict(
            zip(
                ["mask_spk{}".format(i + 1) for i in range(self.num_spk)],
                [complex_mask],
            )
        )

        if self.masking_mode == "E":
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
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

        masked = []
        # shape (B, T, F)
        if is_torch_1_9_plus and self.use_builtin_complex:
            masked.append(torch.complex(real.permute(0, 2, 1), imag.permute(0, 2, 1)))
        else:
            masked.append(ComplexTensor(real.permute(0, 2, 1), imag.permute(0, 2, 1)))

        return (masked, ilens, others)

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    @property
    def num_spk(self):
        return 1
