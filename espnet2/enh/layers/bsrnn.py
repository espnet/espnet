from itertools import accumulate

import torch
import torch.nn as nn


class BSRNN(nn.Module):
    # ported from https://github.com/sungwon23/BSRNN
    def __init__(
        self, input_dim=481, num_channel=16, num_layer=6, target_fs=48000, causal=True
    ):
        """Band-Split RNN (BSRNN).

        References:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim (int): maximum number of frequency bins corresponding to
                `target_fs`
            num_channel (int): embedding dimension of each time-frequency bin
            num_layer (int): number of time and frequency RNN layers
            target_fs (int): maximum sampling frequency supported by the model
            causal (bool): Whether or not to adopt causal processing
                if True, LSTM will be used instead of BLSTM for time modeling
        """
        super().__init__()
        self.num_layer = num_layer
        self.band_split = BandSplit(
            input_dim, target_fs=target_fs, channels=num_channel
        )
        self.target_fs = target_fs
        self.causal = causal

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hdim = 2 * num_channel
        for i in range(self.num_layer):
            self.norm_time.append(nn.GroupNorm(1, num_channel))
            self.rnn_time.append(
                nn.LSTM(
                    num_channel,
                    hdim,
                    batch_first=True,
                    bidirectional=not causal,
                )
            )
            self.fc_time.append(nn.Linear(hdim if causal else hdim * 2, num_channel))
            self.norm_freq.append(nn.GroupNorm(1, num_channel))
            self.rnn_freq.append(
                nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(4 * num_channel, num_channel))

        self.mask_decoder = MaskDecoder(
            input_dim, self.band_split.subbands, channels=num_channel
        )

    def forward(self, x, fs=None):
        """BSRNN forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            out (torch.Tensor): output tensor of shape (B, T, F, 2)
        """
        z = self.band_split(x, fs=fs)
        B, N, T, K = z.shape
        skip = z
        for i in range(self.num_layer):
            out = self.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, T, N)
            out, _ = self.rnn_time[i](out)
            out = self.fc_time[i](out)
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

            out = self.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B * T, K, N)
            out, _ = self.rnn_freq[i](out)
            out = self.fc_freq[i](out)
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = self.mask_decoder(skip)
        m = torch.view_as_complex(m)
        r = torch.view_as_complex(r)
        x = torch.view_as_complex(x)
        m = m[..., : x.size(-1)]
        r = r[..., : x.size(-1)]
        return torch.view_as_real(m * x + r)


class BandSplit(nn.Module):
    def __init__(self, input_dim, target_fs=48000, channels=128):
        super().__init__()
        assert input_dim % 2 == 1, input_dim
        n_fft = (input_dim - 1) * 2
        # freq resolution = target_fs / n_fft = freqs[1] - freqs[0]
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs)
        if input_dim == 481 and target_fs == 48000:
            # n_fft=960 (20ms)
            # first 20 200Hz subbands: [0-200], (200-400], (400-600], ..., (3800-4000]
            # subsequent 6 500Hz subbands: (4000, 4500], ..., (6500, 7000]
            # subsequent 7 2kHz subbands: (7000, 9000], ..., (19000, 21000]
            # final 3kHz subband: (21000, 24000]
            self.subbands = tuple([5] + [4] * 19 + [10] * 6 + [40] * 7 + [60])
        else:
            raise NotImplementedError(
                f"Please define your own subbands for input_dim={input_dim} and "
                f"target_fs={target_fs}"
            )
        self.subband_freqs = freqs[[idx - 1 for idx in accumulate(self.subbands)]]

        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(len(self.subbands)):
            self.norm.append(nn.GroupNorm(1, int(self.subbands[i] * 2)))
            self.fc.append(nn.Conv1d(int(self.subbands[i] * 2), channels, 1))

    def forward(self, x, fs=None):
        """BandSplit forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            z (torch.Tensor): output tensor of shape (B, N, T, K')
                K' might be smaller than len(self.subbands) if fs < self.target_fs.
        """
        if fs is not None:
            assert x.size(2) == sum(self.subbands), (x.size(2), sum(self.subbands))
        hz_band = 0
        for i, subband in enumerate(self.subbands):
            if fs is not None and self.subband_freqs[i] > fs / 2:
                break
            if fs is None and hz_band >= x.size(2):
                break
            x_band = x[:, :, hz_band : hz_band + int(subband), :]
            if int(subband) > x_band.size(2):
                x_band = nn.functional.pad(
                    x_band, (0, 0, 0, int(subband) - x_band.size(2))
                )
            x_band = x_band.reshape(x_band.size(0), x_band.size(1), -1)
            out = self.norm[i](x_band.transpose(1, 2))
            # (B, band * 2, T) -> (B, N, T)
            out = self.fc[i](out)

            if i == 0:
                z = out.unsqueeze(-1)
            else:
                z = torch.cat((z, out.unsqueeze(-1)), dim=-1)
            hz_band = hz_band + int(subband)
        return z


class MaskDecoder(nn.Module):
    def __init__(self, freq_dim, subbands, channels=128):
        super().__init__()
        assert freq_dim == sum(subbands), (freq_dim, subbands)
        self.subbands = subbands
        self.freq_dim = freq_dim
        self.mlp_mask = nn.ModuleList()
        self.mlp_residual = nn.ModuleList()
        for subband in self.subbands:
            self.mlp_mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4), 1),
                    nn.GLU(dim=1),
                )
            )
            self.mlp_residual.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4), 1),
                    nn.GLU(dim=1),
                )
            )

    def forward(self, x):
        """MaskDecoder forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, N, T, K)
        Returns:
            m (torch.Tensor): output mask of shape (B, T, F, 2)
            r (torch.Tensor): output residual of shape (B, T, F, 2)
        """
        for i in range(len(self.subbands)):
            if i >= x.size(-1):
                break
            x_band = x[:, :, :, i]
            out = self.mlp_mask[i](x_band).transpose(1, 2).contiguous()
            out = out.reshape(out.size(0), out.size(1), -1, 2)
            if i == 0:
                m = out
            else:
                m = torch.cat((m, out), dim=2)

            res = self.mlp_residual[i](x_band).transpose(1, 2).contiguous()
            res = res.reshape(res.size(0), res.size(1), -1, 2)
            if i == 0:
                r = res
            else:
                r = torch.cat((r, res), dim=2)
        # Pad zeros in addition to efffective subbands to cover the full frequency range
        m = nn.functional.pad(m, (0, 0, 0, int(self.freq_dim - m.size(-2))))
        r = nn.functional.pad(r, (0, 0, 0, int(self.freq_dim - r.size(-2))))
        return m, r
