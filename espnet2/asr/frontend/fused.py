from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend


class FusedFrontends(AbsFrontend):
    """
        A class that combines multiple frontend modules for feature extraction in speech processing.

    This class allows the fusion of different frontend modules, such as DefaultFrontend and S3prlFrontend,
    and aligns their outputs using a specified method (currently only linear projection is supported).

    Attributes:
        align_method (str): The method used to align and fuse frontend outputs.
        proj_dim (int): The dimension of the projection applied to each frontend's output.
        frontends (torch.nn.ModuleList): A list of frontend modules to be combined.
        gcd (int): The greatest common divisor of hop lengths across all frontends.
        factors (list): A list of factors used for reshaping frontend outputs.
        projection_layers (torch.nn.ModuleList): Linear projection layers for each frontend.

    Args:
        frontends (list): A list of dictionaries, each containing configuration for a frontend.
        align_method (str, optional): The method used to align frontend outputs. Defaults to "linear_projection".
        proj_dim (int, optional): The dimension of the projection applied to each frontend's output. Defaults to 100.
        fs (int, optional): The sampling frequency of the input audio. Defaults to 16000.

    Raises:
        NotImplementedError: If an unsupported frontend type is specified.

    Note:
        Currently, only 'default' and 's3prl' frontend types are supported.
        The 'linear_projection' is the only supported alignment method at the moment.

    Examples:
        >>> frontends = [
        ...     {"frontend_type": "default", "n_mels": 80},
        ...     {"frontend_type": "s3prl", "frontend_conf": "wav2vec2_base"}
        ... ]
        >>> fused_frontend = FusedFrontends(frontends=frontends, proj_dim=128)
    """

    @typechecked
    def __init__(
        self, frontends=None, align_method="linear_projection", proj_dim=100, fs=16000
    ):
        super().__init__()
        self.align_method = (
            align_method  # fusing method : linear_projection only for now
        )
        self.proj_dim = proj_dim  # dim of the projection done on each frontend
        self.frontends = []  # list of the frontends to combine

        for i, frontend in enumerate(frontends):
            frontend_type = frontend["frontend_type"]
            if frontend_type == "default":
                n_mels, fs, n_fft, win_length, hop_length = (
                    frontend.get("n_mels", 80),
                    fs,
                    frontend.get("n_fft", 512),
                    frontend.get("win_length"),
                    frontend.get("hop_length", 128),
                )
                window, center, normalized, onesided = (
                    frontend.get("window", "hann"),
                    frontend.get("center", True),
                    frontend.get("normalized", False),
                    frontend.get("onesided", True),
                )
                fmin, fmax, htk, apply_stft = (
                    frontend.get("fmin", None),
                    frontend.get("fmax", None),
                    frontend.get("htk", False),
                    frontend.get("apply_stft", True),
                )

                self.frontends.append(
                    DefaultFrontend(
                        n_mels=n_mels,
                        n_fft=n_fft,
                        fs=fs,
                        win_length=win_length,
                        hop_length=hop_length,
                        window=window,
                        center=center,
                        normalized=normalized,
                        onesided=onesided,
                        fmin=fmin,
                        fmax=fmax,
                        htk=htk,
                        apply_stft=apply_stft,
                    )
                )
            elif frontend_type == "s3prl":
                frontend_conf, download_dir, multilayer_feature = (
                    frontend.get("frontend_conf"),
                    frontend.get("download_dir"),
                    frontend.get("multilayer_feature"),
                )
                self.frontends.append(
                    S3prlFrontend(
                        fs=fs,
                        frontend_conf=frontend_conf,
                        download_dir=download_dir,
                        multilayer_feature=multilayer_feature,
                    )
                )

            else:
                raise NotImplementedError  # frontends are only default or s3prl

        self.frontends = torch.nn.ModuleList(self.frontends)

        self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
        self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        if self.align_method == "linear_projection":
            self.projection_layers = [
                torch.nn.Linear(
                    in_features=frontend.output_size(),
                    out_features=self.factors[i] * self.proj_dim,
                )
                for i, frontend in enumerate(self.frontends)
            ]
            self.projection_layers = torch.nn.ModuleList(self.projection_layers)
            self.projection_layers = self.projection_layers.to(torch.device(dev))

    def output_size(self) -> int:
        """
                Returns the output size of the fused frontends.

        This method calculates the total output size of all combined frontends
        after projection and fusion.

        Returns:
            int: The total output size, which is the product of the number of
                 frontends and the projection dimension (proj_dim).

        Example:
            >>> fused_frontend = FusedFrontends(frontends=[...], proj_dim=100)
            >>> output_size = fused_frontend.output_size()
            >>> print(output_size)
            200  # Assuming two frontends are used
        """
        return len(self.frontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Processes the input through all frontends and fuses their outputs.

        This method applies each frontend to the input, projects the outputs,
        aligns them, and concatenates them to produce a single fused feature representation.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, num_samples).
            input_lengths (torch.Tensor): Tensor of input lengths for each sample in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - input_feats (torch.Tensor): The fused features from all frontends.
                  Shape: (batch_size, num_frames, total_proj_dim).
                - feats_lens (torch.Tensor): The lengths of the feature sequences for each sample in the batch.

        Raises:
            NotImplementedError: If an unsupported alignment method is specified.

        Note:
            Currently, only the 'linear_projection' alignment method is implemented.

        Example:
            >>> fused_frontend = FusedFrontends(frontends=[...])
            >>> input_tensor = torch.randn(32, 16000)  # (batch_size, num_samples)
            >>> input_lengths = torch.full((32,), 16000)
            >>> fused_features, feature_lengths = fused_frontend.forward(input_tensor, input_lengths)
        """
        # step 0 : get all frontends features
        self.feats = []
        for frontend in self.frontends:
            with torch.no_grad():
                input_feats, feats_lens = frontend.forward(input, input_lengths)
            self.feats.append([input_feats, feats_lens])

        if (
            self.align_method == "linear_projection"
        ):  # TODO(Dan): to add other align methods
            # first step : projections
            self.feats_proj = []
            for i, frontend in enumerate(self.frontends):
                input_feats = self.feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))

            # 2nd step : reshape
            self.feats_reshaped = []
            for i, frontend in enumerate(self.frontends):
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
