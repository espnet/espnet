import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torchaudio
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
        save_dir: str = None,
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
        if save_dir is not None:
            model = AutoModel.from_pretrained(
                download_path, trust_remote_code=True, cache_dir=save_dir
            )
        else:
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


class EnCodecFrontend(AbsFrontend):
    def __init__(
        self,
        fs: Union[int, str] = 48000,
        bandwidth: Union[int, str] = 12,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_path: str = None,
        multilayer_feature: bool = False,
        layer: int = -1,
    ):
        from transformers import AutoProcessor, EncodecModel

        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs not in [48000, 24000]:
            logging.warning(
                "All the Codec models in MERT now only support 48/24 kHz audio."
            )
        self.fs = fs
        model = EncodecModel.from_pretrained(download_path)
        model.eval()
        processor = AutoProcessor.from_pretrained(download_path)
        if layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"
        else:
            layer_selections = None

        self.multilayer_feature = multilayer_feature
        self.layer = layer - 1
        self.model = model
        self.processor = processor
        self.pretrained_params = copy.deepcopy(self.model.state_dict())
        self.frontend_type = "encodec"
        self.tile_factor = 1
        if fs == 48000:
            self.type = "stereo"
        elif fs == 32000:
            self.type = "mono"
        self.stride = 320
        if isinstance(bandwidth, str):
            bandwidth = humanfriendly.parse_size(bandwidth)
        self.bandwidth = bandwidth
        # For the 24 kHz model, supported bandwidths are 1.5kbps (n_q = 2),
        # 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
        # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
        # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
        self.n_q = {
            24000: {1.5: 2, 3: 4, 6: 8, 12: 16, 24: 32},
            48000: {3: 2, 6: 4, 12: 8, 24: 16},
        }
        assert self.bandwidth in self.n_q[self.fs], "Bandwidth not supported."
        assert (
            self.layer < self.n_q[self.fs][self.bandwidth]
        ), "For {}kps, n_q = {}".format(
            self.bandwidth, self.n_q[self.fs][self.bandwidth]
        )

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
        if self.type == "stereo":
            if len(input.shape) == 2:
                input = input.unsqueeze(1).expand(-1, 2, -1)
            assert input.size(1) == 2
        elif self.type == "mono":
            if len(input.shape) == 2:
                input = input.unsqueeze(1)
            assert input.size(1) == 1
        # feats = [[] for i in range(self.n_q[self.fs][self.bandwidth])]
        hs = []
        feats_lens = []
        for i in range(len(input)):
            inputs = self.processor(
                raw_audio=input[i][:, : input_lengths[i]].cpu(),
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt",
            ).to(input.device)
            out = self.model.encode(
                inputs["input_values"], inputs["padding_mask"], bandwidth=self.bandwidth
            )
            codes = out.audio_codes
            scales = out.audio_scales
            codes = torch.cat([c[0] for c in codes], dim=-1)
            hs.append(codes)
            feats_lens.append(codes.size(1))

        feats = torch.zeros(
            [self.n_q[self.fs][self.bandwidth], len(input), max(feats_lens)]
        ).to(input.device)
        for i in range(self.n_q[self.fs][self.bandwidth]):
            for j in range(len(input)):
                feats[i][j][: len(hs[j][i])] = hs[j][i]
        feats = feats.long()
        feats_lens = torch.tensor(feats_lens).to(input.device)
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
        logging.info("Pretrained EnCodec frontend model parameters reloaded!")
