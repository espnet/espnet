import copy
import logging
from typing import List, Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.legacy.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.tasks.ssl import SSLTask
from espnet2.utils.get_default_kwargs import get_default_kwargs


class ESPnetSSLFrontend(AbsFrontend):
    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        masking_conf: Optional[dict] = {},
        multilayer_feature: bool = False,
        layer: int = -1,
        freeze_encoder_steps: int = 0,
        mask_feats: bool = True,
        use_final_output: bool = True,
    ):
        """Frontend wrapper for SSL models trained in ESPnet.

        Args:
            fs (int): unused.
            frontend_conf (dict): make sure path_or_url has a value
            multilayer_feature (bool): whether to use weighted sum
            layer (int): 0-indexed layer to use if not using weighted sum
            mask_feats (int): whether to mask input feats to encoder
            use_final_output (bool): use final normalized output instead of
                last layer output. For post-LN model archs.
        """
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning(
                f"SSL models usually only 16 kHz audio, make sure fs={fs} is intended."
            )

        # Build model from URL or local path
        # Assume local models end with .pt or .pth
        path_or_url = frontend_conf.get("path_or_url", None)
        if path_or_url is not None:
            if path_or_url.endswith(".pt") or path_or_url.endswith(".pth"):
                model, model_args = SSLTask.build_model_from_file(
                    None,
                    path_or_url,
                    device="cpu",
                )
            else:
                try:
                    from espnet_model_zoo.downloader import ModelDownloader

                except ImportError:
                    logging.error(
                        "`espnet_model_zoo` is not installed. "
                        "Please install via `pip install -U espnet_model_zoo`."
                    )
                    raise

                args = {}
                d = ModelDownloader()
                args.update(**d.download_and_unpack(path_or_url))
                model, model_args = SSLTask.build_model_from_file(
                    args["ssl_train_config"],
                    args["ssl_model_file"],
                    device="cpu",
                )
        else:
            raise RuntimeError("Did not receive a model to load in the frontend_conf")

        if multilayer_feature:
            featurizer = Featureizer(len(model.encoder.encoders))
        else:
            featurizer = torch.nn.Identity()

        model.train()

        # remove unneeded pre-training modules for DDP
        del model.losses
        if "ema" in model.util_attributes:
            del model.util_modules["ema"]

        if "mask" in model.util_modules:
            for key in masking_conf:
                if hasattr(model.util_modules["mask"], key):
                    setattr(model.util_modules["mask"], key, masking_conf[key])

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.upstream = model
        self.featurizer = featurizer
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "espnet_ssl"
        self.mask_feats = mask_feats
        self.use_final_output = use_final_output

        self.register_buffer("global_step", torch.tensor([0]))
        self.freeze_encoder_steps = freeze_encoder_steps

    def output_size(self) -> int:
        return self.upstream.encoder.output_size()

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            self.global_step = self.global_step + 1

        if self.training and self.mask_feats:
            use_mask = True
        else:
            use_mask = False

        if self.global_step <= self.freeze_encoder_steps:
            with torch.no_grad():
                input, layer_outputs, input_lengths = self.upstream.inference_encode(
                    input, input_lengths, use_mask, self.use_final_output
                )
        else:
            input, layer_outputs, input_lengths = self.upstream.inference_encode(
                input, input_lengths, use_mask, self.use_final_output
            )

        if self.multilayer_feature:
            return self.featurizer(layer_outputs), input_lengths
        else:
            return layer_outputs[self.layer], input_lengths

    def reload_pretrained_parameters(self):
        self.upstream.load_state_dict(self.pretrained_params)
        del self.pretrained_params
        self.upstream.train()
        logging.info("Pretrained ESPNet frontend model parameters reloaded!")


class Featureizer(torch.nn.Module):
    def __init__(self, num_layers: int):
        """Simplified S3PRL-style featurizer.

        Outputs a learned weighted sum of input layers.
        Original code by Leo Yang (2022) in the S3PRL library.
        https://github.com/s3prl/s3prl/blob/main/s3prl/nn/upstream.py
        """
        super().__init__()

        self.layer_selections = list(range(num_layers))
        self.weights = torch.nn.Parameter(torch.zeros(len(self.layer_selections)))

    def _weighted_sum(self, all_hs):
        stacked_hs = torch.stack(all_hs, dim=0)

        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(len(self.layer_selections), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs

    def forward(self, all_hs: List[torch.FloatTensor]):
        """Forward function.

        Args:
            all_hs (List[torch.FloatTensor]): List[ (batch_size, seq_len, hidden_size) ]

        Returns:
            1. The weighted-sum result, (batch_size, seq_len, hidden_size)
        """
        if len(all_hs) == 1:
            return all_hs[0]

        all_hs = [h for idx, h in enumerate(all_hs) if idx in self.layer_selections]
        hs = self._weighted_sum(all_hs)
        return hs
