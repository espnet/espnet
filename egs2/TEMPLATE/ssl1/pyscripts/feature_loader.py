# -*- coding: utf-8 -*-
# The feature_loader.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_mfcc_feature.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

"""Extract MFCC & intermediate embedding from the Hubert model for k-means clustering."""

import logging
import os
import sys

import fairseq

import soundfile as sf
import torch
import torchaudio

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("feature_loader")


class MfccFeatureReader(object):
    def __init__(self, fs):
        self.fs = fs

    def load_audio(self, path):
        wav, sr = sf.read(path)
        assert sr == self.fs, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        return wav

    def get_feats(self, path):
        x = self.load_audio(path)
        with torch.no_grad():
            x = torch.from_numpy(x).view(1, -1).float()

            mfcc = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.fs,
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


class HubertFeatureReader(object):
    def __init__(self, fs, hubert_url, hubert_dir_path, layer, max_chunk=1600000):
        self.fs = fs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder

        e = FairseqHubertEncoder(0, hubert_url, hubert_dir_path)
        self.model = e.encoders.to(self.device).eval()

        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def load_audio(self, path):
        wav, sr = sf.read(path)
        assert sr == self.fs, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        return wav

    def get_feats(self, path):
        x = self.load_audio(path)
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
