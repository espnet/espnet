from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend


class FusedFrontends(AbsFrontend):
    """
    A class to fuse multiple audio frontends for feature extraction.

    This class combines multiple audio frontends, such as `DefaultFrontend`
    and `S3prlFrontend`, into a single module. It allows for the alignment and
    projection of features extracted from these frontends using a specified
    method. Currently, only linear projection is supported for fusing the
    frontends.

    Attributes:
        align_method (str): The method used for aligning features. Currently,
            only "linear_projection" is supported.
        proj_dim (int): The dimension of the projection applied to each
            frontend's output.
        frontends (ModuleList): A list of frontends to combine.
        gcd (int): The greatest common divisor of the hop lengths of the
            frontends.
        factors (list): The factors for reshaping the output based on hop lengths.
        projection_layers (ModuleList): A list of linear layers for projecting
            frontend outputs.

    Args:
        frontends (list): A list of dictionaries specifying the frontends to
            combine. Each dictionary should include the type of frontend and
            its respective parameters.
        align_method (str, optional): The alignment method for feature fusion.
            Defaults to "linear_projection".
        proj_dim (int, optional): The dimension for projection. Defaults to 100.
        fs (int, optional): The sampling frequency. Defaults to 16000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The fused feature tensor and the
            lengths of the features.

    Raises:
        NotImplementedError: If an unsupported frontend type is provided or if
            an unsupported alignment method is specified.

    Examples:
        # Example of initializing FusedFrontends with Default and S3PRL frontends
        frontends_config = [
            {"frontend_type": "default", "n_mels": 80, "n_fft": 512},
            {"frontend_type": "s3prl", "frontend_conf": {}, "download_dir": ""}
        ]
        fused_frontend = FusedFrontends(frontends=frontends_config)

        # Forward pass through the fused frontend
        input_tensor = torch.randn(10, 16000)  # Example input tensor
        input_lengths = torch.tensor([16000] * 10)  # Example lengths
        output_feats, output_lengths = fused_frontend(input_tensor, input_lengths)

    Note:
        The class is currently limited to using the linear projection alignment
        method. Future implementations may include additional alignment methods.
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
        Calculates the output size of the fused frontends based on the number of
        frontends and the projection dimension.

        The output size is determined by multiplying the number of frontends by the
        projection dimension specified during initialization. This value is useful
        for determining the shape of the output tensor after processing the input
        through the fused frontends.

        Returns:
            int: The total output size of the fused frontends.

        Examples:
            >>> fused_frontend = FusedFrontends(
            ...     frontends=[
            ...         {"frontend_type": "default", "n_mels": 80},
            ...         {"frontend_type": "s3prl", "frontend_conf": {...}},
            ...     ],
            ...     proj_dim=100
            ... )
            >>> output_size = fused_frontend.output_size()
            >>> print(output_size)
            200  # (2 frontends * 100 proj_dim)

        Note:
            The function assumes that the `frontends` attribute is properly initialized
            and contains valid frontend configurations.
        """
        return len(self.frontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass for the FusedFrontends class, which processes
        input audio through multiple frontends and aligns the output features
        based on the specified alignment method.

        Args:
            input (torch.Tensor): The input audio tensor of shape (batch_size,
                num_samples).
            input_lengths (torch.Tensor): A tensor containing the lengths of the
                input sequences of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of fused audio features of shape
                (batch_size, num_frames, output_size).
                - A tensor of the lengths of the output features of shape
                (batch_size,).

        Raises:
            NotImplementedError: If the alignment method is not supported.

        Examples:
            >>> fused_frontend = FusedFrontends(frontends=[{"frontend_type": "default"}])
            >>> input_tensor = torch.randn(10, 16000)  # 10 samples of 1 second each
            >>> input_lengths = torch.tensor([16000] * 10)  # lengths for each sample
            >>> output_feats, output_lengths = fused_frontend.forward(input_tensor, input_lengths)
            >>> print(output_feats.shape)  # Expected output shape: (10, num_frames, output_size)

        Note:
            The current implementation supports only the 'linear_projection'
            alignment method. Future updates may include additional methods.
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
