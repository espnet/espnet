# -*- coding: utf-8 -*-
# The feature_loader.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_mfcc_feature.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

"""Extract MFCC & intermediate embedding from the Hubert model for k-means clustering"""

import logging

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("feature_loader")


class BaseFeatureReader(object):
    def __init__(self):
        raise NotImplementedError

    def load_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, data, ref_len=None):
        raise NotImplementedError

class ESPnetModelFeatureReader(BaseFeatureReader):
    def __init__(self, asr_model_path, layer=None, sample_rate=16000, max_chunk=1600000):
        self.sample_rate = sample_rate

        device = "cuda" if torch.cuda.is_available() else "cpu"
        from espnet2.tasks.asr import ASRTask

        asr_model, asr_train_args = ASRTask.build_model_from_file(
            None,
            asr_model_path,
            device,
        )

        self.device = next(asr_model.parameters()).device
        self.model = asr_model.eval()
        self.layer = layer

        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def get_ssl_weights(self):
        
        if self.model.frontend is not None and self.model.frontend.featurizer is not None:
            weights = self.model.frontend.featurizer.weights
            norm_weights = F.softmax(weights, dim=-1)
            labels = range(len(norm_weights))
            sizes = norm_weights.detach().numpy() * 100
            sizes = sizes.tolist()
            return sizes, labels

    def get_feats(self, data, ref_len=None):
        if isinstance(data, str):
            x = self.load_audio(data, ref_len=ref_len)
        elif isinstance(data, np.ndarray):
            x = data
        with torch.inference_mode():
            x = torch.from_numpy(x).float().to(self.device)
            x = x.view(1, -1) # torch.Size([1, 32640])

            lens = torch.tensor([x.shape[1]], dtype=torch.long)
            feat, _ = self.model.encode(x, lens, layer=self.layer)

            # feat is (feat, inter_feat) for models using interctc
            if type(feat) is tuple:
                feat = feat[0]

            feat = feat[-1]  # (time, feat_dim) torch.Size([49, 512])
        return feat.cpu()
