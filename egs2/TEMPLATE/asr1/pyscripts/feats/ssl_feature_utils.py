import json
import logging
import os
import re
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.dataset import ESPnetDataset
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


def build_data_iterator(
    rspecifier: str,
    in_filetype: str,
    utt2num_samples: str,
    batch_bins: Optional[int] = 1,
):
    dataset = ESPnetDataset(
        [(rspecifier[4:], "speech", in_filetype)],
        preprocess=None,
    )
    sampler = NumElementsBatchSampler(
        batch_bins=batch_bins,
        shape_files=[utt2num_samples],
    )
    batches = list(sampler)
    iterator = SequenceIterFactory(
        dataset=dataset,
        batches=batches,
        collate_fn=CommonCollateFn(float_pad_value=0.0, int_pad_value=-1),
        num_workers=2,
    ).build_iter(0)
    return iterator


def dump_feature(
    reader,
    in_filetype: str,
    rspecifier: str,
    out_filetype: str,
    wspecifier: str,
    utt2num_samples: Optional[str] = None,
    batch_bins: Optional[int] = None,
    write_num_frames: bool = None,
):
    assert os.path.exists(utt2num_samples), f"{utt2num_samples} does not exist."

    iterator = build_data_iterator(rspecifier, in_filetype, utt2num_samples, batch_bins)

    with file_writer_helper(
        wspecifier,
        filetype=out_filetype,
        write_num_frames=write_num_frames,
    ) as writer:
        for utt_ids, data in iterator:
            feats, feats_lens = reader.get_feats(data["speech"], data["speech_lengths"])
            for idx, utt in enumerate(utt_ids):
                writer[utt] = feats[idx][: feats_lens[idx]].numpy()
    logger.info("finished successfully")


class BaseFeatureReader(object):
    def __init__(self):
        raise NotImplementedError

    def load_audio(self, path: str, ref_len: Optional[int] = None):
        wav, sr = sf.read(path)
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def preprocess_data(
        self,
        data: Union[str, np.ndarray, list, torch.Tensor],
        data_lens: Union[int, List[int], torch.Tensor],
        ref_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data, data_lens
        elif isinstance(data, str):
            batch_size = 1
            x = self.load_audio(data, ref_len=ref_len)
        elif isinstance(data, np.ndarray):
            batch_size = 1
            x = data
        else:
            raise TypeError(f"Unexpected data type of argument 1: {type(data)}.")
        x = torch.from_numpy(x).view(batch_size, -1).float()
        x_lens = torch.tensor([data_lens]).long()
        return x, x_lens

    def get_feats(
        self, data: torch.Tensor, data_lens: torch.Tensor, ref_len: Optional[int] = None
    ):
        raise NotImplementedError


class MfccFeatureReader(BaseFeatureReader):
    def __init__(
        self,
        sample_rate: int = 16000,
        **kwargs,  # placeholder for unused arguments
    ):
        self.sample_rate = sample_rate
        self.frame_length = 25 * sample_rate / 1000
        self.frame_shift = 10 * sample_rate / 1000

    def get_feats(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        ref_len: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        feats, feats_lens = [], []
        with torch.no_grad():
            x, x_lens = self.preprocess_data(data, data_lens)
            batch_size = x.shape[0]
            for i in range(batch_size):
                mfcc = torchaudio.compliance.kaldi.mfcc(
                    waveform=x[i : i + 1],
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
                feats.append(concat)
                feats_lens.append(
                    int((x_lens[i] - self.frame_length) // self.frame_shift + 1)
                )
        return feats, feats_lens


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

    def get_feats(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        ref_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x, x_lens = self.preprocess_data(data, data_lens)
            x = x.to(self.device)
            mask = x.zeros_like(x, dtype=torch.long)
            for i in range(x.shape[0]):
                mask[i, x_lens[i] :].fill_(1)

            feats, feats_padding_mask = [], []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, feat_mask = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=mask[:, start : start + self.max_chunk],
                    mask=False,
                    output_layer=self.layer,
                )
                feats.append(feat_chunk)
                feats_padding_mask.append(feat_mask)

        feats = torch.cat(feats, 1).cpu()
        feats_padding_mask = torch.cat(feats_padding_mask, 1).cpu()
        feats_lens = (1 - feats_padding_mask).sum(dim=1)
        return feats, feats_lens


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
        self.model = hubert_model.encoder.hubert_pretrain_model.to(self.device).eval()

        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def get_feats(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        ref_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            x, x_lens = self.preprocess_data(data, data_lens)
            x = x.to(self.device)
            x_lens = x_lens.to(self.device)

            feats, feats_lens = self.model.wav2vec2.extract_features(
                waveforms=x,
                lengths=x_lens,
                num_layers=self.layer,
            )
            feats = feats[-1].cpu()  # (batchsize, time, feat_dim)
        return feats, feats_lens


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

    def get_feats(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        ref_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x, x_lens = self.preprocess_data(data, data_lens)
            x = x.to(self.device)

            feats, feats_lens = self.model(x, x_lens)
        feats = feats.cpu()
        feats_lens = feats_lens.cpu()
        return feats, feats_lens
