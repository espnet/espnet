import logging
import os
import sys

import numpy as np
import torch

from hubert_feature_loader import BaseFeatureReader


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("wavlm_feature_loader")


class WavLMFeatureReader(BaseFeatureReader):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        sys.path.append("./local/wavlm")
        from WavLM import WavLM, WavLMConfig

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint["model"])
        self.model = model.eval().cuda()
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def get_feats(self, data, ref_len=None):
        if isinstance(data, str):
            x = self.load_audio(data, ref_len=ref_len)
        elif isinstance(data, np.ndarray):
            x = data
        else:
            raise TypeError(f"Unexpected data type of argument 1: {type(data)}.")
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda().unsqueeze(0)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
            return torch.cat(feat, 1).squeeze(0)
