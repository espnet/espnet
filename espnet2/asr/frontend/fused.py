import copy
from typing import Optional
from typing import Tuple
from typing import Union
from argparse import Namespace
import logging
import os
import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.nets_utils import pad_list

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend


class FusedFrontends(AbsFrontend):

    def __init__(
            self,
            n_mels: int = 80,
            frontends = None ,
            align_method = "linear_projection",
            proj_dim = 100,
            fs="16k"
    ):

        assert check_argument_types()
        super().__init__()

        self.n_mels = n_mels
        self.align_method = align_method
        self.proj_dim = proj_dim

        self.frontends = []
        for i,frontend in enumerate(frontends) :
            print(i)
            frontend_type = frontend["frontend_type"]
            if frontend_type == "default":
                n_mels, fs, n_fft, win_length, hop_length = n_mels, fs, frontend["n_fft"], frontend["win_length"], frontend["hop_length"]
               # window, center , normalized , onesided = frontend["window"], frontend["center"], frontend["normalized"], frontend["onesided"]
                self.frontends.append(DefaultFrontend(n_mels=n_mels, n_fft=n_fft, fs=fs, win_length=win_length, hop_length=hop_length))
            elif frontend_type == "s3prl":
                frontend_conf = frontend["frontend_conf"]
                download_dir = frontend["download_dir"]
                multilayer_feature = frontend["multilayer_feature"]
                self.frontends.append(S3prlFrontend(fs=fs, frontend_conf=frontend_conf, download_dir=download_dir, multilayer_feature=multilayer_feature))
            else :
                raise NotImplementedError

        self.frontends = torch.nn.ModuleList(self.frontends)

        self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
        self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]

        print("ok gcd", self.gcd)

        if self.align_method == "linear_projection":
            self.projection_layers = [torch.nn.Linear(in_features=frontend.output_size(),
                                                          out_features=self.factors[i] * self.proj_dim, device="cuda") for i,frontend in enumerate(self.frontends)]

        print("ok align")


    def output_size_default(self) -> int:
        return self.n_mels

    def output_size_s3prl(self) -> int:
        return self.output_dim_s3prl

    def output_size(self) -> int:
        return sum([f.output_size() for f in self.frontends])




    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.feats = []
        for frontend in self.frontends :
            with torch.no_grad():
                input_feats, feats_lens = frontend.forward(input, input_lengths)
            self.feats.append([input_feats, feats_lens])


        if self.align_method == "linear_projection":

            # first step : projections
            self.feats_proj = []
            for i, frontend in enumerate(self.frontends) :
                input_feats = self.feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))

            # 2nd step : reshape
            self.feats_reshaped = []
            for i, frontend in enumerate(self.frontends):
                input_feats_proj = self.feats_proj[i]
                bs, nf, dim = input_feats_proj.shape
                input_feats_reshaped = torch.reshape(input_feats_proj, (bs, nf * self.factors[i], dim // self.factors[i]))
                self.feats_reshaped.append(input_feats_reshaped)

            # 3rd step : drop the few last frames
            m = min([x.shape[1] for x in self.feats_reshaped])
            self.feats_final = [x[:, :m, :] for x in self.feats_reshaped]


            input_feats = torch.cat(self.feats_final, dim=-1)  # change the input size of the preencoder in conf file : proj_dim * n_frontends
            feats_lens = torch.ones_like(self.feats[0][1]) * (m)


        else:
            raise NotImplementedError

        return input_feats, feats_lens


