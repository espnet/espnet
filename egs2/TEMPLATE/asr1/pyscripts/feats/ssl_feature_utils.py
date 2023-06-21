import json
import logging
import os
import re
import sys
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import is_scipy_wav_style
from espnet.utils.cli_writers import file_writer_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("s3prl_feature_loader")


def format_feature_conf_str(feature_conf: str):
    # 1. removing any extraneous white spaces
    feature_conf = re.sub(r"\s", "", feature_conf)
    # Surrounding any word/path with "
    feature_conf = re.sub(r"([\w\.\-/]+)", r'"\1"', feature_conf)
    # Replacing = with :
    feature_conf = re.sub(r"=", ": ", feature_conf)
    try:
        feature_conf = json.loads(feature_conf)
    except Exception as e:
        logger.warning(f"Failure in parsing feature_conf {feature_conf}")
        raise e
    return feature_conf


def dump_feature(
    reader, in_filetype, rspecifier, out_filetype, wspecifier, write_num_frames=None
):
    with file_writer_helper(
        wspecifier,
        filetype=out_filetype,
        write_num_frames=write_num_frames,
    ) as writer:
        for utt, mat in file_reader_helper(rspecifier, in_filetype):
            if is_scipy_wav_style(mat):
                # If data is sound file, then got as Tuple[int, ndarray]
                rate, mat = mat
                mat = mat.astype(np.float64, order="C") / 32768.0
            nsample = len(mat)
            feat = reader.get_feats(mat, nsample).numpy()
            writer[utt] = feat
    logger.info("finished successfully")


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
    def __init__(
        self,
        sample_rate=16000,
        **kwargs,  # placeholder for unused arguments
    ):
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
        self,
        hubert_url,
        hubert_dir_path,
        layer,
        sample_rate=16000,
        max_chunk=1600000,
        use_gpu=True,
    ):
        self.sample_rate = sample_rate

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
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
    def __init__(
        self,
        hubert_model_path,
        layer,
        sample_rate=16000,
        max_chunk=1600000,
        use_gpu=True,
    ):
        self.sample_rate = sample_rate

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        from espnet2.tasks.hubert import HubertTask

        hubert_model, hubert_train_args = HubertTask.build_model_from_file(
            None,
            hubert_model_path,
            self.device,
        )
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

            feat = self.model.wav2vec2.extract_features(
                x,
                num_layers=self.layer,
            )[
                0
            ][-1][
                0
            ]  # (time, feat_dim)
        return feat.cpu()


class S3PRLFeatureReader(BaseFeatureReader):
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        s3prl_conf: Optional[dict] = None,
        download_dir: str = None,
        multilayer_feature: bool = False,
        layer: int = -1,
        use_gpu: bool = True,
    ):
        self.model = S3prlFrontend(
            fs=fs,
            frontend_conf=s3prl_conf,
            download_dir=download_dir,
            multilayer_feature=multilayer_feature,
            layer=layer,
        )
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def get_feats(self, data: Union[str, np.ndarray], ref_len=None):
        if isinstance(data, str):
            x = self.load_audio(data, ref_len=ref_len)
        elif isinstance(data, np.ndarray):
            x = data
        else:
            raise TypeError(f"Unexpected data type of argument 1: {type(data)}.")

        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            x = x.view(1, -1)

            feat, _ = self.model(x, torch.LongTensor([ref_len]))
            return feat.squeeze(0).cpu()
