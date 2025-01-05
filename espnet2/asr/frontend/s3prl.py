import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlFrontend(AbsFrontend):
    """
    Speech Pretrained Representation frontend structure for ASR.

    This class implements a frontend that utilizes S3PRL to extract
    pretrained speech representations for automatic speech recognition (ASR).
    It allows for configurable parameters, including the sample rate, the choice
    of upstream model, and whether to use multilayer features.

    Attributes:
        multilayer_feature (bool): Flag indicating whether to use multilayer
            features or not.
        layer (int): Specific layer to extract features from. If -1, the last
            layer is used.
        upstream: The upstream model used for feature extraction.
        featurizer: The featurizer that processes the upstream outputs.
        pretrained_params (dict): A copy of the pretrained parameters for
            reloading.
        frontend_type (str): The type of frontend, set to "s3prl".
        hop_length (int): The hop length of the feature extraction.
        tile_factor (int): The factor by which to tile the representations.

    Args:
        fs (Union[int, str]): The sample rate of the audio input. Defaults to
            16000 Hz.
        frontend_conf (Optional[dict]): Configuration dictionary for the
            frontend. It should include the upstream model name and other
            parameters. Defaults to the default configuration of Frontend.
        download_dir (Optional[str]): Directory to download models if needed.
            Defaults to None.
        multilayer_feature (bool): Whether to use features from multiple layers.
            Defaults to False.
        layer (int): The specific layer to extract features from. Defaults to
            -1 (last layer).

    Raises:
        Exception: If S3PRL is not installed correctly or if an invalid upstream
            model is specified.

    Examples:
        >>> frontend = S3prlFrontend(fs=16000, multilayer_feature=True)
        >>> input_tensor = torch.randn(1, 16000)  # Example input tensor
        >>> input_lengths = torch.tensor([16000])  # Example input length
        >>> features, lengths = frontend(input_tensor, input_lengths)
        >>> print(features.shape)  # Shape of the extracted features

    Note:
        This class requires S3PRL to be installed. If not installed, an error
        message will be printed with installation instructions.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: Optional[str] = None,
        multilayer_feature: bool = False,
        layer: int = -1,
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer, S3PRLUpstream
        except Exception as e:
            print("Error: S3PRL is not properly installed.")
            print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
            raise e

        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning(
                "All the upstream models in S3PRL now only support 16 kHz audio."
            )

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert frontend_conf.get("upstream", None) in S3PRLUpstream.available_names()
        upstream = S3PRLUpstream(
            frontend_conf.get("upstream"),
            path_or_url=frontend_conf.get("path_or_url", None),
            normalize=frontend_conf.get("normalize", False),
            extra_conf=frontend_conf.get("extra_conf", None),
        )
        if getattr(upstream.upstream, "model", None):
            if getattr(upstream.upstream.model, "feature_grad_mult", None) is not None:
                upstream.upstream.model.feature_grad_mult = 1.0
        upstream.eval()

        if layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"
        else:
            layer_selections = None
        featurizer = Featurizer(upstream, layer_selections=layer_selections)

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.upstream, self.featurizer = upstream, featurizer
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "s3prl"
        self.hop_length = self.featurizer.downsample_rate
        self.tile_factor = frontend_conf.get("tile_factor", 1)

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        """
        Get the output size of the featurizer.

        This method returns the output size, which corresponds to the number
        of features produced by the featurizer component of the S3PRL frontend.

        Returns:
            int: The output size of the featurizer.

        Examples:
            >>> frontend = S3prlFrontend()
            >>> output_size = frontend.output_size()
            >>> print(output_size)
            512  # This value may vary based on the upstream model configuration.

        Note:
            The output size is determined by the specific upstream model used
            and its configuration. Make sure the featurizer has been properly
            initialized before calling this method.
        """
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the S3prlFrontend class.

        This method processes the input tensor and its lengths through the
        upstream model to extract features. Depending on the configuration,
        it can return features from a specific layer, multiple layers, or apply
        tiling to the representations.

        Args:
            input (torch.Tensor): Input tensor containing audio data. The shape
                should be (batch_size, seq_len, feature_dim).
            input_lengths (torch.Tensor): Lengths of the input sequences. The
                shape should be (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feats (torch.Tensor): The extracted features after processing
                  through the upstream model.
                - feats_lens (torch.Tensor): The lengths of the extracted
                  features.

        Examples:
            >>> input_tensor = torch.randn(32, 16000)  # 32 samples of 1 second
            >>> input_lengths = torch.tensor([16000] * 32)
            >>> frontend = S3prlFrontend()
            >>> features, lengths = frontend.forward(input_tensor, input_lengths)

        Note:
            The input audio data should be sampled at 16 kHz for compatibility
            with the upstream models.

        Raises:
            AssertionError: If the input feature shape is not (batch_size,
            seq_len, feature_dim).
        """
        feats, feats_lens = self.upstream(input, input_lengths)
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return feats, feats_lens

    def reload_pretrained_parameters(self):
        """
        Reload the pretrained parameters of the S3PRL frontend model.

        This method allows the user to restore the model's parameters to their
        pretrained state, which can be useful for resetting the model or for
        experiments where the pretrained parameters need to be re-applied.

        It retrieves the parameters stored in `self.pretrained_params` and loads
        them back into the upstream model. After successfully reloading, a log
        message is generated to confirm the action.

        Examples:
            >>> frontend = S3prlFrontend()
            >>> # Assuming some training or fine-tuning has been done here
            >>> frontend.reload_pretrained_parameters()
            Pretrained S3PRL frontend model parameters reloaded!

        Note:
            Ensure that the `reload_pretrained_parameters` method is called
            when the model is in a valid state and after it has been initialized
            with pretrained parameters.

        Raises:
            RuntimeError: If the model's state cannot be reloaded due to
            incompatible shapes or other issues related to the model's architecture.
        """
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
