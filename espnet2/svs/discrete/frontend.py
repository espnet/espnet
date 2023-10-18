import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class MERTFrontend(AbsFrontend):
    def __init__(
        self,
        fs: Union[int, str] = 24000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_path: str = None,
        multilayer_feature: bool = False,
        layer: int = -1,
    ):
        from transformers import AutoModel

        assert check_argument_types()
        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 24000:
            logging.warning(
                "All the RVQ-based models in MERT now only support 24 kHz audio."
            )
        model = AutoModel.from_pretrained(download_path, trust_remote_code=True)
        model.eval()

        if layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"
        else:
            layer_selections = None

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.model = model
        self.pretrained_params = copy.deepcopy(self.model.state_dict())
        self.frontend_type = "mert"
        self.tile_factor = 1
        self.stride = 320

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
        return self.model.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_mask = make_non_pad_mask(input_lengths)
        feats = self.model(
            input_values=input, attention_mask=input_mask, output_hidden_states=True
        ).hidden_states
        feats_lens = (
            torch.div(input_lengths - 1, self.stride, rounding_mode="floor") + 1
        )
        feats = [h[:, : max(feats_lens), :] for h in feats]
        feats_lens = [feats_lens] * len(feats)
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return feats, feats_lens

    def reload_pretrained_parameters(self):
        self.model.load_state_dict(self.pretrained_params)
        logging.info("Pretrained MERT frontend model parameters reloaded!")
