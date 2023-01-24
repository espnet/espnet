# -*- coding: utf-8 -*-
# The feature_loader.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_mfcc_feature.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import logging

import numpy as np
import soundfile as sf
import torch
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


class MfccFeatureReader(BaseFeatureReader):
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def get_feats(self, data, ref_len=None):
        if isinstance(data, str):
            x = self.load_audio(data, ref_len=ref_len)
        elif isinstance(data, np.ndarray):
            x = data
        else:
            raise TypeError(f"Unexpected data type of argument 1: {type(data)}.")
        with torch.no_grad():
            x = torch.from_numpy(x).view(1, -1).float()

            mfcc = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
            ).transpose(
                0, 1
            )  # (freq, time)
            delta = torchaudio.functional.compute_deltas(mfcc)
            ddelta = torchaudio.functional.compute_deltas(delta)
            concat = (
                torch.cat([mfcc, delta, ddelta], dim=0).transpose(0, 1).contiguous()
            )
            return concat


class HubertFeatureReader(BaseFeatureReader):
    def __init__(
        self, hubert_url, hubert_dir_path, layer, sample_rate=16000, max_chunk=1600000
    ):
        self.sample_rate = sample_rate

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder

        e = FairseqHubertEncoder(0, hubert_url, hubert_dir_path)
        self.model = e.encoders.to(self.device).eval()

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
            x = torch.from_numpy(x).float().to(self.device)
            x = x.view(1, -1)

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
            return torch.cat(feat, 1).squeeze(0).cpu()


class ESPnetHubertFeatureReader(BaseFeatureReader):
    def __init__(self, hubert_model_path, layer, sample_rate=16000, max_chunk=1600000):
        self.sample_rate = sample_rate

        device = "cuda" if torch.cuda.is_available() else "cpu"
        from espnet2.tasks.hubert import HubertTask

        hubert_model, hubert_train_args = HubertTask.build_model_from_file(
            None,
            hubert_model_path,
            device,
        )
        self.device = next(hubert_model.parameters()).device
        self.model = hubert_model.encoder.hubert_pretrain_model.eval()

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
        with torch.inference_mode():
            x = torch.from_numpy(x).float().to(self.device)
            x = x.view(1, -1)

            feat = self.model.wav2vec2.extract_features(x, num_layers=self.layer,)[0][
                -1
            ][
                0
            ]  # (time, feat_dim)
        return feat.cpu()


class S3PRLFeatureReader(BaseFeatureReader):
    def __init__(self, s3prl_upstream_name, layer, sample_rate=16000, max_chunk=1600000):
        self.sample_rate = sample_rate
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from s3prl.nn import S3PRLUpstream
        except ModuleNotFoundError:
            S3PRLUpstream = None
            raise RuntimeError(
                "cannot find s3prl, please install s3prl via tools/installers"
            )
        
        self.model = S3PRLUpstream(s3prl_upstream_name).to(self.device)
        self.model.eval()

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
        with torch.inference_mode():
            x = torch.from_numpy(x).float().to(self.device)
            x = x.view(1, -1)
            wav_len = torch.LongTensor([x.size(1)])
            all_hs, all_hs_lens = self.model(x, wav_len)
            feat = all_hs[self.layer][0]

        return feat.cpu()
