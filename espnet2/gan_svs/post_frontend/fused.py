from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend  # noqa
from espnet2.gan_svs.post_frontend.s3prl import S3prlPostFrontend


class FusedPostFrontends(AbsFrontend):
    @typechecked
    def __init__(
        self,
        postfrontends=None,
        align_method="linear_projection",
        proj_dim=100,
        fs=16000,
        input_fs=24000,
    ):
        super().__init__()
        self.align_method = (
            align_method  # fusing method : linear_projection only for now
        )
        self.proj_dim = proj_dim  # dim of the projection done on each postfrontend
        self.postfrontends = []  # list of the postfrontends to combine

        for i, postfrontend in enumerate(postfrontends):
            postfrontend_type = postfrontend["postfrontend_type"]

            if postfrontend_type == "s3prl":
                postfrontend_conf, download_dir, multilayer_feature = (
                    postfrontend.get("postfrontend_conf"),
                    postfrontend.get("download_dir"),
                    postfrontend.get("multilayer_feature"),
                )
                self.postfrontends.append(
                    S3prlPostFrontend(
                        fs=fs,
                        input_fs=input_fs,
                        postfrontend_conf=postfrontend_conf,
                        download_dir=download_dir,
                        multilayer_feature=multilayer_feature,
                    )
                )
            else:
                raise NotImplementedError  # frontends are only s3prl

        self.postfrontends = torch.nn.ModuleList(self.postfrontends)

        self.gcd = np.gcd.reduce(
            [postfrontend.hop_length for postfrontend in self.postfrontends]
        )
        self.factors = [
            postfrontend.hop_length // self.gcd for postfrontend in self.postfrontends
        ]

        if self.align_method == "linear_projection":
            self.projection_layers = [
                torch.nn.Linear(
                    in_features=postfrontend.output_size(),
                    out_features=self.factors[i] * self.proj_dim,
                )
                for i, postfrontend in enumerate(self.postfrontends)
            ]
            self.projection_layers = torch.nn.ModuleList(self.projection_layers)

    def output_size(self) -> int:
        return len(self.postfrontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # step 0 : get all frontends features
        self.feats = []
        for postfrontend in self.postfrontends:
            with torch.no_grad():
                input_feats, feats_lens = postfrontend.forward(input, input_lengths)
            self.feats.append([input_feats, feats_lens])

        if self.align_method == "linear_projection":
            # first step : projections
            self.feats_proj = []
            for i, postfrontend in enumerate(self.postfrontends):
                input_feats = self.feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))

            # 2nd step : reshape
            self.feats_reshaped = []
            for i, postfrontend in enumerate(self.postfrontends):
                input_feats_proj = self.feats_proj[i]
                bs, nf, dim = input_feats_proj.shape
                input_feats_reshaped = torch.reshape(
                    input_feats_proj, (bs, nf * self.factors[i], dim // self.factors[i])
                )
                self.feats_reshaped.append(input_feats_reshaped)

            # 3rd step : drop the few last frames
            m = min([x.shape[1] for x in self.feats_reshaped])
            self.feats_final = [x[:, :m, :] for x in self.feats_reshaped]

            input_feats = torch.cat(
                self.feats_final, dim=-1
            )  # change the input size of the preencoder : proj_dim * n_frontends
            feats_lens = torch.ones_like(self.feats[0][1]) * (m)

        else:
            raise NotImplementedError

        return input_feats, feats_lens
