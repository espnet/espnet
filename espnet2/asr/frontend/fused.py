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

from s3prl import base_s3prl_setup
import s3prl
import default

class Fused_Frontends(AbsFrontend):

    def __init__(
            self,
            fs: Union[int, str] = 16000,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            window: Optional[str] = "hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True,
            n_mels: int = 80,
            fmin: int = None,
            fmax: int = None,
            htk: bool = False,
            frontend_conf: Union[Optional[dict]]= [get_default_kwargs(Frontend), get_default_kwargs(Frontend)] ,
            apply_stft: bool = True,
            download_dir: Union[str] = [None, None] , # put None for default, but always fill a value
            multilayer_feature: Union[bool] = [False, False] ,  # put False for default, but always fill a value
            align_method: str = "linear_projection",
            frontends: list = ["default","s3prl"],
            proj_dim: int = 100
    ):
# a revoir pcq chacun des s3prl va avoir une dim differente, et des params differents, faut faire que tous
# les arguments pour s3prl soient des listes ducoup, pour default pas besoin pcq yen a que un au plus
        assert check_argument_types()
        super().__init__()

        self.n_mels = n_mels
        self.align_method = align_method

        self.frontends = []
        for i,frontend in enumerate(frontends) :
            if frontend == "default":
                self.frontends.append(default.DefaultFrontend(fs,n_fft,win_length,hop_length,window,center,normalized,onesided,n_mels,fmin,fmax,htk,frontend_conf[i],apply_stft))
            elif frontend == "s3prl":
                self.frontends.append(s3prl.S3prlFrontend(fs,frontend_conf[i],download_dir[i],multilayer_feature[i]))
            else :
                raise NotImplementedError


        self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
        self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]

        self.proj_dim = proj_dim

        if self.align_method == "linear_projection":
            self.projection_layers = [torch.nn.Linear(in_features=frontend.output_size(),
                                                          out_features=self.factor[i] * self.proj_dim) for i,frontend in enumerate(self.frontends)]



    def output_size_default(self) -> int:
        return self.n_mels

    def output_size_s3prl(self) -> int:
        return self.output_dim_s3prl

    def output_size(self) -> int:
        return self.output_size_default() + self.output_size_s3prl()



    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.feats = []
        with torch.no_grad():
            for frontend in self.frontends :
            input_feats, feats_lens = frontend.forward(input, input_lengths)
            self.feats.append([input_feats, feats_lens])


        if self.align_method == "linear_projection":

            # first step : projections
            self.feats_proj = []
            for i, frontend in enumerate(self.frontends) :
                input_feats = self.feats[i]
                self.feats_proj.append(self.projection_layers[i](input_feats))

            # 2nd step : reshape
            self.feats_reshaped = []
            for i, frontend in enumerate(self.frontends):
                input_feats_proj = self.feats_proj[i]
                bs, nf, dim = input_feats_proj.shape
                input_feats_reshaped = torch.reshape(input_feats_proj, (bs, nf * self.factors[i], dim // self.factor[i]))
                self.feats_reshaped.append(input_feats_reshaped)

            # 3rd step : drop the few last frames
            m = min([x.shape[1] for x in self.feats_reshaped])
            self.feats_final = [x[:, :m, :] for x in self.feats_reshaped]


            input_feats = torch.cat(self.feats_final, dim=-1)  # change the input size of the preencoder in conf file : proj_dim * n_frontends
            feats_lens = torch.ones_like(self.feats[0][1]) * (m)


        else:
            raise NotImplementedError

        return input_feats, feats_lens


