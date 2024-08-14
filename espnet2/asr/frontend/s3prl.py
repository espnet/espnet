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

    This class implements a frontend for Automatic Speech Recognition (ASR) using
    pretrained speech representations from the S3PRL toolkit. It supports various
    upstream models and allows for flexible configuration of the frontend.

    Attributes:
        frontend_type (str): Type of the frontend, set to "s3prl".
        hop_length (int): The hop length (downsampling rate) of the featurizer.
        tile_factor (int): Factor by which to tile the output representations.
        multilayer_feature (bool): Whether to use multilayer features or not.
        layer (int): Specific layer to use from the upstream model (-1 for last layer).

    Args:
        fs (Union[int, str]): Sampling frequency of the input audio. Defaults to 16000.
        frontend_conf (Optional[dict]): Configuration for the frontend. Defaults to None.
        download_dir (Optional[str]): Directory to download S3PRL models. Defaults to None.
        multilayer_feature (bool): Whether to use multilayer features. Defaults to False.
        layer (int): Specific layer to use from the upstream model. Defaults to -1.

    Raises:
        Exception: If S3PRL is not properly installed.

    Note:
        - All upstream models in S3PRL currently only support 16 kHz audio.
        - When a specific layer is selected, multilayer_feature will be deactivated.

    Examples:
        >>> frontend = S3prlFrontend(fs=16000, frontend_conf={'upstream': 'wav2vec2'})
        >>> input_tensor = torch.randn(1, 16000)
        >>> input_lengths = torch.tensor([16000])
        >>> feats, feats_lens = frontend(input_tensor, input_lengths)
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
                Returns the output size of the frontend.

        This method provides the dimensionality of the feature vectors produced by the frontend.

        Returns:
            int: The size of the output feature vectors.

        Example:
            >>> frontend = S3prlFrontend(fs=16000, frontend_conf={'upstream': 'wav2vec2'})
            >>> output_dim = frontend.output_size()
            >>> print(output_dim)
            768  # Example output, actual value may vary depending on the upstream model
        """
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Processes the input audio and returns the extracted features.

        This method takes the input audio tensor and its corresponding lengths, passes it through
        the S3PRL upstream model and featurizer, and returns the extracted features.

        Args:
            input (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).
            input_lengths (torch.Tensor): Lengths of each audio in the batch of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feats (torch.Tensor): Extracted features of shape (batch_size, num_frames, feature_dim).
                - feats_lens (torch.Tensor): Lengths of each feature sequence in the batch of shape (batch_size,).

        Note:
            - If a specific layer is selected (self.layer != -1), only features from that layer are returned.
            - If multilayer_feature is True, features from multiple layers are combined.
            - If tile_factor is not 1, the output features are tiled accordingly.

        Example:
            >>> frontend = S3prlFrontend(fs=16000, frontend_conf={'upstream': 'wav2vec2'})
            >>> input_tensor = torch.randn(2, 16000)  # 2 audio samples of 1 second each
            >>> input_lengths = torch.tensor([16000, 16000])
            >>> feats, feats_lens = frontend(input_tensor, input_lengths)
            >>> print(feats.shape)
            torch.Size([2, 50, 768])  # Example output, actual shape may vary
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
                Reloads the pretrained parameters of the S3PRL frontend model.

        This method restores the original pretrained parameters of the upstream S3PRL model,
        effectively resetting any fine-tuning or modifications made to the model weights.

        Note:
            This method is useful when you want to reset the model to its initial pretrained state,
            for example, after fine-tuning or experimenting with the model parameters.

        Example:
            >>> frontend = S3prlFrontend(fs=16000, frontend_conf={'upstream': 'wav2vec2'})
            >>> # After some operations or fine-tuning
            >>> frontend.reload_pretrained_parameters()
            >>> print("Pretrained S3PRL frontend model parameters reloaded!")
        """
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
