# Implementation of TCN-DenseUNet in
# Wang, Zhong-Qiu, Gordon Wichern, and Jonathan Le Roux.
# "Leveraging Low-Distortion Target Estimates for Improved Speech Enhancement."
# arXiv preprint arXiv:2110.00570 (2021).


import torch
import asteroid_filterbanks.transforms as af_transforms
from asteroid.masknn import activations


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 num_freqs,
                  ksz=3,
                 activation=torch.nn.ELU, hid_chans=32):
        super(DenseBlock, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hid_chans, (ksz, ksz), (1, 1), "same"),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(in_channels),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels + hid_chans, hid_chans, (ksz, ksz), (1, 1), "same"),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(in_channels),
        )

        self.bn_freq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels + hid_chans * 2, hid_chans, (1, 1), (1, 1), "same"
            ),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(in_channels),
        )

        self.freq_processing = torch.nn.Sequential(
            torch.nn.Conv2d(num_freqs, num_freqs, (1, 1), (1, 1), "same"),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(in_channels),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels + hid_chans * 3, hid_chans, (ksz, ksz), (1, 1), "same"
            ),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(in_channels),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels + hid_chans * 4, out_channels, (ksz, ksz), (1, 1), "same"
            ),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(in_channels),
        )

    def forward(self, input):

        out1 = self.conv1(input)
        out2 = self.conv2(torch.cat((out1, input), 1))
        out3 = self.freq_processing(
            self.bn_freq(torch.cat((out2, out1, input), 1)).permute(0, 3, 2, 1)
        ).permute(0, 3, 2, 1)
        out4 = self.conv3(torch.cat((out3, out2, out1, input), 1))
        out5 = self.conv4(torch.cat((out4, out3, out2, out1, input), 1))

        return out5


class TCNResBlock(torch.nn.Module):
    def __init__(self, d=384, ksz=3, dilation=1, activation=torch.nn.ELU):
        super(TCNResBlock, self).__init__()
        padding = dilation
        self.layer = torch.nn.Sequential(
            torch.nn.InstanceNorm1d(d),
            activations.get(activation)(),
            torch.nn.Conv1d(d, d, ksz, 1, padding, dilation, groups=d),
        )

    def forward(self, inp):
        return self.layer(inp) + inp


class TCNDenseUNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        hid_chans=32,
        ksz_dense=3,
        ksz_tcn=3,
        tcn_repeats=4,
        tcn_blocks=7,
        tcn_channels=384,
        activation=torch.nn.ELU
    ):
        super(TCNDenseUNet, self).__init__()
        self.in_channels = in_channels
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels*2, hid_chans, (3, 3), (1, 1), (1, 0)))
        self.enc1 = DenseBlock(hid_chans, hid_chans, 255, ksz_dense, activation)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.enc2 = DenseBlock(hid_chans, hid_chans, 127, ksz_dense, activation)

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.enc3 = DenseBlock(hid_chans, hid_chans, 63, ksz_dense, activation)

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.enc4 = DenseBlock(hid_chans, hid_chans, 31,  ksz_dense, activation)

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.enc5 = DenseBlock(hid_chans, hid_chans, 15,  ksz_dense, activation)

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans, hid_chans * 2, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans * 2, hid_chans * 4, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans * 4),
        )
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(hid_chans * 4, tcn_channels, (3, 3), (1, 1), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(tcn_channels),
        )

        self.tcn = []
        for r in range(tcn_repeats):
            for x in range(tcn_blocks):
                self.tcn.append(TCNResBlock(tcn_channels, ksz_tcn, 2 ** x))

        self.tcn = torch.nn.Sequential(*self.tcn)

        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                tcn_channels * 2, hid_chans * 4, (3, 3), (1, 1), (1, 0)
            ), activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans * 4),
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                hid_chans * 8, hid_chans * 2, (3, 3), (1, 2), (1, 0)
            ), activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans * 2),
        )
        self.dconv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hid_chans * 4, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )

        self.dec5 = torch.nn.Sequential(
            DenseBlock(hid_chans * 2, hid_chans*2, 15, ksz_dense, activation),
            torch.nn.ConvTranspose2d(hid_chans*2, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.dec4 = torch.nn.Sequential(
            DenseBlock(hid_chans * 2, hid_chans*2, 31, ksz_dense, activation),
            torch.nn.ConvTranspose2d(hid_chans*2, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.dec3 = torch.nn.Sequential(
            DenseBlock(hid_chans * 2, hid_chans*2, 63, ksz_dense, activation),
            torch.nn.ConvTranspose2d(hid_chans*2, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )
        self.dec2 = torch.nn.Sequential(
            DenseBlock(hid_chans * 2, hid_chans*2, 127, ksz_dense, activation),
            torch.nn.ConvTranspose2d(hid_chans*2, hid_chans, (3, 3), (1, 2), (1, 0)),
            activations.get(activation)(),
            torch.nn.InstanceNorm2d(hid_chans),
        )

        self.dec1 = torch.nn.Sequential(
            DenseBlock(hid_chans * 2, hid_chans*2, 255, ksz_dense, activation),
            torch.nn.ConvTranspose2d(hid_chans*2, 2, (3, 3), (1, 1), (1, 0)),
        )

    def forward(self, tf_rep):

        bsz, mics, _, frames = tf_rep.shape
        inp_feats = af_transforms.to_torch_complex(tf_rep)
        inp_feats = torch.cat((inp_feats.real, inp_feats.imag), 1)
        inp_feats = inp_feats.transpose(-1, -2)
        inp_feats = inp_feats.reshape(bsz, self.in_channels * 2, frames, -1)

        enc1 = self.enc1(self.conv1(inp_feats))
        enc2 = self.enc2(self.conv2(enc1))
        enc3 = self.enc3(self.conv3(enc2))
        enc4 = self.enc4(self.conv4(enc3))
        enc5 = self.enc5(self.conv5(enc4))
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)
        assert enc8.shape[-1] == 1
        out = self.tcn(enc8.squeeze(-1))
        out = self.dconv8(torch.cat((out.unsqueeze(-1), enc8), 1))
        out = self.dconv7(torch.cat((out, enc7), 1))
        out = self.dconv6(torch.cat((out, enc6), 1))
        out = self.dec5(torch.cat((out, enc5), 1))
        out = self.dec4(torch.cat((out, enc4), 1))
        out = self.dec3(torch.cat((out, enc3), 1))
        out = self.dec2(torch.cat((out, enc2), 1))
        out = self.dec1(torch.cat((out, enc1), 1))

        out = torch.cat((out[:, 0], out[:, 1]), -1)
        return out