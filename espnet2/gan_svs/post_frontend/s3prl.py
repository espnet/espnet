import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torchaudio
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlPostFrontend(AbsFrontend):
    """
        S3prlPostFrontend is a pretrained SSL model for VISinger2 Plus. It is based on
    the S3prlFrontend and adds a resampler to resample the input audio to the
    sample rate of the pretrained model.

    Attributes:
        fs (int): The target sample rate for the model (default: 16000).
        input_fs (int): The input audio sample rate (default: 24000).
        multilayer_feature (bool): Flag to indicate if multilayer features are used
            (default: False).
        layer (int): The specific layer to extract features from (default: -1).
        upstream (S3PRLUpstream): The upstream model used for feature extraction.
        featurizer (Featurizer): The featurizer that processes the upstream model's
            output.
        pretrained_params (dict): A copy of the upstream model's state dictionary.
        frontend_type (str): Type of the frontend, set to "s3prl".
        hop_length (int): The hop length used in the feature extraction.
        tile_factor (int): The factor by which to tile the representations.
        resampler (torchaudio.transforms.Resample): Resampler for audio input.

    Args:
        fs (Union[int, str]): Target sample rate (default: 16000).
        input_fs (Union[int, str]): Input audio sample rate (default: 24000).
        postfrontend_conf (Optional[dict]): Configuration dictionary for the
            postfrontend (default: None).
        download_dir (Optional[str]): Directory to download pretrained models
            (default: None).
        multilayer_feature (bool): Flag to indicate if multilayer features are
            extracted (default: False).
        layer (int): Specific layer to extract features from (default: -1).

    Raises:
        ImportError: If S3PRL is not properly installed.

    Examples:
        # Initialize S3prlPostFrontend with default parameters
        s3prl_frontend = S3prlPostFrontend()

        # Initialize with custom parameters
        custom_frontend = S3prlPostFrontend(fs=22050, input_fs=44100,
                                             multilayer_feature=True)

    Note:
        The upstream models in S3PRL currently only support 16 kHz audio.

    Todo:
        - Consider adding support for different upstream model sample rates.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        input_fs: Union[int, str] = 24000,
        postfrontend_conf: Optional[dict] = get_default_kwargs(Frontend),
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

        if isinstance(input_fs, str):
            input_fs = humanfriendly.parse_size(input_fs)

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert (
            postfrontend_conf.get("upstream", None) in S3PRLUpstream.available_names()
        )
        upstream = S3PRLUpstream(
            postfrontend_conf.get("upstream"),
            path_or_url=postfrontend_conf.get("path_or_url", None),
            normalize=postfrontend_conf.get("normalize", False),
            extra_conf=postfrontend_conf.get("extra_conf", None),
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
        self.tile_factor = postfrontend_conf.get("tile_factor", 1)
        self.resampler = torchaudio.transforms.Resample(orig_freq=input_fs, new_freq=fs)
        self.fs = fs
        self.input_fs = input_fs

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
        Get the output size of the feature extractor.

        This method retrieves the output size from the featurizer component
        of the S3prlPostFrontend. The output size is determined by the
        configuration of the upstream model used for feature extraction.

        Returns:
            int: The size of the output features produced by the featurizer.

        Examples:
            >>> s3prl_frontend = S3prlPostFrontend()
            >>> output_size = s3prl_frontend.output_size()
            >>> print(output_size)
            512  # Example output size based on the upstream model configuration.
        """
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes input audio through the S3PRL model to extract features.

        This method takes audio input and its corresponding lengths, resamples
        the audio if necessary, and retrieves features from the S3PRL upstream
        model. The features can be extracted from a specific layer or as a
        multi-layer representation depending on the class configuration.

        Args:
            input (torch.Tensor): A tensor containing the input audio data,
                typically of shape (batch_size, num_samples).
            input_lengths (torch.Tensor): A tensor containing the lengths of
                the input audio for each batch element, typically of shape
                (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feats (torch.Tensor): The extracted features from the S3PRL
                  model, shape depends on the configuration.
                - feats_lens (torch.Tensor): The lengths of the extracted
                  features for each batch element, shape (batch_size,).

        Raises:
            AssertionError: If input audio does not meet the expected shape
            or if the layer selection conflicts with multilayer feature setting.

        Examples:
            >>> model = S3prlPostFrontend()
            >>> audio_input = torch.randn(2, 24000)  # Example batch of audio
            >>> input_lengths = torch.tensor([24000, 24000])  # Lengths of inputs
            >>> features, lengths = model.forward(audio_input, input_lengths)

        Note:
            The input audio must be in the sample rate defined by `input_fs`.
            If `fs` (the model's required sample rate) differs from `input_fs`,
            the input audio will be resampled accordingly.
        """
        if self.fs != self.input_fs:
            input = self.resampler(input)
            input_lengths = input_lengths * self.fs / self.input_fs
            input_lengths = input_lengths.ceil().long()

        # You can choose to freeze parameters in the configuration
        # by setting `freeze_param`
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

        This method is useful when you want to reset the model to its original
        pretrained state. It loads the parameters stored in `self.pretrained_params`
        back into the upstream model, allowing for experimentation with different
        initializations or restoring the model after fine-tuning.

        Example:
            >>> model = S3prlPostFrontend()
            >>> model.reload_pretrained_parameters()  # Reloads the original parameters

        Note:
            Ensure that the model is properly initialized before calling this method
            to avoid loading errors.

        Raises:
            RuntimeError: If the model's state_dict does not match the expected
            structure during loading.
        """
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
