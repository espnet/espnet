import copy
import logging
from typing import List, Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlFrontendCondition(AbsFrontend):
    """This is a modified version of S3prlFrontend for geolocation-aware LID.

    For the geolocation-aware LID, please refer to the following paper:
        Geolocation-Aware Robust Spoken Language Identification

    This class requires a modified version of S3PRL with S3PRLUpstreamCondition
    support. Installation:
        git clone -b lid https://github.com/Qingzheng-Wang/s3prl.git
        cd s3prl
        pip install -e .
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: Optional[str] = None,
        multilayer_feature: bool = False,
        layer: Optional[Union[int, List[int]]] = None,
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer, S3PRLUpstreamCondition
        except Exception:
            raise ImportError(
                "Error: s3prl is not found.\n"
                "Please install the modified S3PRL version:\n"
                "  If you have already installed s3prl, please uninstall it first.\n"
                "  (optional) pip uninstall s3prl\n"
                "  git clone -b lid https://github.com/Qingzheng-Wang/s3prl.git\n"
                "  cd s3prl\n"
                "  pip install -e ."
            )

        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning(
                "All the upstream models in S3PRL only support 16 kHz audio."
            )

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert (
            frontend_conf.get("upstream", None)
            in S3PRLUpstreamCondition.available_names()
        ), f"Invalid upstream model: {frontend_conf.get('upstream', None)}"

        upstream = S3PRLUpstreamCondition(
            frontend_conf.get("upstream"),
            path_or_url=frontend_conf.get("path_or_url", None),
            normalize=frontend_conf.get("normalize", False),
            extra_conf=frontend_conf.get("extra_conf", None),
        )
        if getattr(upstream.upstream, "model", None):
            if getattr(upstream.upstream.model, "feature_grad_mult", None) is not None:
                upstream.upstream.model.feature_grad_mult = 1.0
        upstream.eval()

        if layer is not None and isinstance(layer, int):
            # Check that the selected layer index is valid
            assert 0 <= layer < upstream.num_layers, (
                f"Invalid layer index: {layer}, "
                f"should be in [0, {upstream.num_layers - 1}]"
            )
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when a specific layer used"
        elif layer is not None and isinstance(layer, list):
            assert all([0 <= layer_idx < upstream.num_layers for layer_idx in layer]), (
                f"Invalid layer index: {layer}, "
                f"should all be in [0, {upstream.num_layers}]"
            )
            layer_selections = layer
            assert multilayer_feature, (
                "multilayer feature will be activated, "
                "when a list of specific layers used"
            )
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
        ), f"Input argument `feature` has invalid shape: {feature.shape}"
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (feats, feats_lens, intermediate_lang2vec_preds) = self.upstream(
            input, input_lengths, labels
        )
        if self.layer is not None and isinstance(self.layer, int):
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return (feats, feats_lens, intermediate_lang2vec_preds)
