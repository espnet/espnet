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
    """Pretrained SSL model for VISinger2 Plus.

    Based on S3prlFrontend,
    S3prlPostFrontend added a resampler to resample the input audio to
    the sample rate of the pretrained model.
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
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.upstream.load_state_dict(self.pretrained_params)
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")
