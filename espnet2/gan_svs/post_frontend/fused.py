from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.gan_svs.post_frontend.s3prl import S3prlPostFrontend


class FusedPostFrontends(AbsFrontend):
    """
    FusedPostFrontends combines multiple post frontends using specified alignment
    methods. Currently, only the linear projection method is supported for fusing.

    Attributes:
        align_method (str): The method used for aligning features. Default is
            "linear_projection".
        proj_dim (int): The dimension of the projection done on each postfrontend.
            Default is 100.
        postfrontends (torch.nn.ModuleList): A list of the postfrontends to combine.
        gcd (int): The greatest common divisor of hop lengths of all postfrontends.
        factors (list): A list of factors used for reshaping the features from each
            postfrontend.
        projection_layers (torch.nn.ModuleList): A list of linear layers for
            projecting the output of each postfrontend.

    Args:
        postfrontends (list): A list of dictionaries, where each dictionary
            contains the configuration for a postfrontend. Each should specify
            the 'postfrontend_type' and relevant parameters for initialization.
        align_method (str, optional): The method for feature alignment.
            Defaults to "linear_projection".
        proj_dim (int, optional): The dimension of the projected features.
            Defaults to 100.
        fs (int, optional): Sampling frequency for the postfrontends. Defaults to
            16000.
        input_fs (int, optional): Input sampling frequency. Defaults to 24000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the concatenated
        projected features and their lengths.

    Raises:
        NotImplementedError: If an unsupported postfrontend type is specified
            or an unsupported alignment method is requested.

    Examples:
        >>> postfrontends_config = [
        ...     {
        ...         "postfrontend_type": "s3prl",
        ...         "postfrontend_conf": {},
        ...         "download_dir": "/path/to/dir",
        ...         "multilayer_feature": True,
        ...     }
        ... ]
        >>> fused_frontend = FusedPostFrontends(postfrontends=postfrontends_config)
        >>> input_tensor = torch.randn(5, 24000)  # Batch of 5, 24000 samples
        >>> input_lengths = torch.tensor([24000] * 5)  # All inputs are of length 24000
        >>> output_feats, output_lengths = fused_frontend(input_tensor, input_lengths)

    Note:
        This class currently supports only the S3PRL postfrontend type. Future
        implementations may include additional postfrontend types.

    Todo:
        - Implement additional alignment methods beyond linear projection.
    """

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
        """
            A class to fuse multiple post frontends using linear projection.

        This class combines different post frontends, specifically
        S3PRL post frontends, and aligns their outputs using a linear
        projection method. It is designed to facilitate the integration
        of various frontend features into a single representation.

        Attributes:
            align_method (str): The method used for fusing the frontends.
                Currently supports only "linear_projection".
            proj_dim (int): The dimensionality of the projection done on
                each post frontend.
            postfrontends (ModuleList): A list of the post frontends to combine.
            gcd (int): The greatest common divisor of the hop lengths of
                the post frontends.
            factors (list): A list of factors derived from the hop lengths
                of the post frontends.
            projection_layers (ModuleList): A list of linear layers for
                projecting features from each post frontend.

        Args:
            postfrontends (list): A list of dictionaries, each containing
                configurations for the post frontends.
            align_method (str): Method to align features, defaults to
                "linear_projection".
            proj_dim (int): Dimension of the projection, defaults to 100.
            fs (int): Sampling frequency, defaults to 16000.
            input_fs (int): Input sampling frequency, defaults to 24000.

        Returns:
            int: The total output size of the fused features.

        Examples:
            >>> post_frontends = [
            ...     {
            ...         "postfrontend_type": "s3prl",
            ...         "postfrontend_conf": {...},
            ...         "download_dir": "/path/to/download",
            ...         "multilayer_feature": True,
            ...     }
            ... ]
            >>> fused_frontend = FusedPostFrontends(postfrontends=post_frontends)
            >>> output_size = fused_frontend.output_size()
            >>> print(output_size)
            200  # Assuming proj_dim is 100 and there are 2 post frontends

        Note:
            The class currently only supports S3PRL post frontends. Any other
            type will raise a NotImplementedError.

        Raises:
            NotImplementedError: If a post frontend type other than "s3prl"
                is provided.

        Todo:
            - Implement additional alignment methods beyond linear projection.
        """
        return len(self.postfrontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass for the FusedPostFrontends class, which processes input data
        through multiple post-frontends and aligns their features using a specified
        alignment method. Currently, only the linear projection method is supported.

        Args:
            input (torch.Tensor): Input tensor containing the audio features.
            input_lengths (torch.Tensor): Lengths of the input sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor with the concatenated features from all post-frontends after
                  applying the linear projection and reshaping.
                - A tensor with the lengths of the processed features.

        Raises:
            NotImplementedError: If an unsupported alignment method is specified.

        Examples:
            >>> fused_frontend = FusedPostFrontends(postfrontends=[{"postfrontend_type": "s3prl",
            ... "postfrontend_conf": {}, "download_dir": "/path/to/dir",
            ... "multilayer_feature": False}])
            >>> input_tensor = torch.randn(10, 16000)  # Example input tensor
            >>> input_lengths = torch.tensor([16000] * 10)  # Example input lengths
            >>> output_feats, output_lengths = fused_frontend.forward(input_tensor, input_lengths)

        Note:
            Ensure that the input tensor is correctly shaped and that the input lengths
            are accurate to avoid runtime errors during processing.
        """
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
