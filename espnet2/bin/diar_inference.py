#!/usr/bin/env python3

import argparse
from itertools import permutations
import logging
from pathlib import Path
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.functional as F
from tqdm import trange
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import humanfriendly_parse_size_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


class DiarizeSpeech:
    """DiarizeSpeech class

    Examples:
        >>> import soundfile
        >>> diarization = DiarizeSpeech("diar_config.yaml", "diar.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> diarization(audio)
        [(spk_id, start, end), (spk_id2, start2, end2)]

    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        segment_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        normalize_output_wav: bool = False,
        num_spk: Optional[int] = None,
        device: str = "cpu",
        dtype: str = "float32",
        enh_s2t_task: bool = False,
        multiply_diar_result: bool = False,
    ):
        assert check_argument_types()

        task = DiarizationTask if not enh_s2t_task else EnhS2TTask

        # 1. Build Diar model
        diar_model, diar_train_args = task.build_model_from_file(
            train_config, model_file, device
        )
        if enh_s2t_task:
            diar_model.inherite_attributes(
                inherite_s2t_attrs=[
                    "max_num_spk",
                    "decoder",
                    "attractor",
                ]
            )

        diar_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.diar_train_args = diar_train_args
        self.diar_model = diar_model

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = normalize_segment_scale
        self.normalize_output_wav = normalize_output_wav
        self.show_progressbar = show_progressbar
        # not specifying "num_spk" in inference config file
        # will enable speaker number prediction during inference
        self.num_spk = num_spk

        # multiply_diar_result corresponds to the "Post-processing"
        # in https://arxiv.org/pdf/2203.17068.pdf
        self.multiply_diar_result = multiply_diar_result
        self.enh_s2t_task = enh_s2t_task

        self.segmenting_diar = segment_size is not None and not enh_s2t_task
        self.segmenting_enh_diar = (
            segment_size is not None and hop_size is not None and enh_s2t_task
        )
        if self.segmenting_diar:
            logging.info("Perform segment-wise speaker diarization")
            logging.info("Segment length = {} sec".format(segment_size))
        elif self.segmenting_enh_diar:
            logging.info("Perform segment-wise speech separation and diarization")
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speaker diarization on the input")

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], fs: int = 8000
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [speaker_info1, speaker_info2, ...]

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)

        assert speech.dim() > 1, speech.size()
        batch_size = speech.size(0)
        speech = speech.to(getattr(torch, self.dtype))
        # lengths: (B,)
        lengths = speech.new_full(
            [batch_size], dtype=torch.long, fill_value=speech.size(1)
        )

        # a. To device
        speech = to_device(speech, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if self.segmenting_diar and lengths[0] > self.segment_size * fs:
            # Segment-wise speaker diarization
            # Note that the segments are processed independently for now
            # i.e., no speaker tracing is performed
            num_segments = int(np.ceil(speech.size(1) / (self.segment_size * fs)))
            t = T = int(self.segment_size * fs)
            pad_shape = speech[:, :T].shape
            diarized_wavs = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.segment_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech[:, st:en]
                else:
                    t = T
                    speech_seg = speech[:, st:en]  # B x T [x C]

                lengths_seg = speech.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Diarization Forward
                encoder_out, encoder_out_lens = self.encode(
                    speech_seg,
                    lengths_seg,
                )
                spk_prediction, _ = self.decode(encoder_out, encoder_out_lens)
                # List[torch.Tensor(B, T, num_spks)]
                diarized_wavs.append(spk_prediction)
            # Determine maximum estimated number of speakers among the segments
            max_len = max([x.size(2) for x in diarized_wavs])
            # pad tensors in diarized_wavs with "float('-inf')" to have same size
            diarized_wavs = [
                torch.nn.functional.pad(
                    x, (0, max_len - x.size(2)), "constant", float("-inf")
                )
                for x in diarized_wavs
            ]
            spk_prediction = torch.cat(diarized_wavs, dim=1)
        else:
            # b. Diarization Forward
            encoder_out, encoder_out_lens = self.encode(speech, lengths)
            spk_prediction, num_spk = self.decode(encoder_out, encoder_out_lens)
            if self.enh_s2t_task:
                # Segment-wise speech separation
                # Note that this is done after diarization using the whole sequence
                if self.segmenting_enh_diar and lengths[0] > self.segment_size * fs:
                    overlap_length = int(
                        np.round(fs * (self.segment_size - self.hop_size))
                    )
                    num_segments = int(
                        np.ceil(
                            (speech.size(1) - overlap_length) / (self.hop_size * fs)
                        )
                    )
                    t = T = int(self.segment_size * fs)
                    pad_shape = speech[:, :T].shape
                    enh_waves = []
                    range_ = trange if self.show_progressbar else range
                    for i in range_(num_segments):
                        st = int(i * self.hop_size * fs)
                        en = st + T
                        if en >= lengths[0]:
                            # en - st < T (last segment)
                            en = lengths[0]
                            speech_seg = speech.new_zeros(pad_shape)
                            t = en - st
                            speech_seg[:, :t] = speech[:, st:en]
                        else:
                            t = T
                            speech_seg = speech[:, st:en]  # B x T [x C]

                        lengths_seg = speech.new_full(
                            [batch_size], dtype=torch.long, fill_value=T
                        )
                        # Separation Forward
                        _, _, processed_wav = self.diar_model.encode_diar(
                            speech_seg, lengths_seg, num_spk
                        )
                        if self.normalize_segment_scale:
                            # normalize the scale to match the input mixture scale
                            mix_energy = torch.sqrt(
                                torch.mean(
                                    speech_seg[:, :t].pow(2), dim=1, keepdim=True
                                )
                            )
                            enh_energy = torch.sqrt(
                                torch.mean(
                                    sum(processed_wav)[:, :t].pow(2),
                                    dim=1,
                                    keepdim=True,
                                )
                            )
                            processed_wav = [
                                w * (mix_energy / enh_energy) for w in processed_wav
                            ]
                        # List[torch.Tensor(num_spk, B, T)]
                        enh_waves.append(torch.stack(processed_wav, dim=0))

                    # c. Stitch the enhanced segments together
                    waves = enh_waves[0]
                    for i in range(1, num_segments):
                        # permutation between separated streams in last and current segments
                        perm = self.cal_permumation(
                            waves[:, :, -overlap_length:],
                            enh_waves[i][:, :, :overlap_length],
                            criterion="si_snr",
                        )
                        # repermute separated streams in current segment
                        for batch in range(batch_size):
                            enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                        if i == num_segments - 1:
                            enh_waves[i][:, :, t:] = 0
                            enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                        else:
                            enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                        # overlap-and-add (average over the overlapped part)
                        waves[:, :, -overlap_length:] = (
                            waves[:, :, -overlap_length:]
                            + enh_waves[i][:, :, :overlap_length]
                        ) / 2
                        # concatenate the residual parts of the later segment
                        waves = torch.cat([waves, enh_waves_res_i], dim=2)
                    # ensure the stitched length is same as input
                    assert waves.size(2) == speech.size(1), (waves.shape, speech.shape)
                    waves = torch.unbind(waves, dim=0)
                else:
                    # Separation Forward using the whole signal
                    _, _, waves = self.diar_model.encode_diar(speech, lengths, num_spk)
                # multiply diarization result and separation result
                # by calculating the correlation
                if self.multiply_diar_result:
                    spk_prediction, interp_prediction, _ = self.permute_diar(
                        waves, spk_prediction
                    )
                    waves = [
                        waves[i] * interp_prediction[:, :, i] for i in range(num_spk)
                    ]
                if self.normalize_output_wav:
                    waves = [
                        (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                        for w in waves
                    ]  # list[(batch, sample)]
                else:
                    waves = [w.cpu().numpy() for w in waves]

        if self.num_spk is not None:
            assert spk_prediction.size(2) == self.num_spk, (
                spk_prediction.size(2),
                self.num_spk,
            )
        assert spk_prediction.size(0) == batch_size, (
            spk_prediction.size(0),
            batch_size,
        )
        spk_prediction = spk_prediction.cpu().numpy()
        spk_prediction = 1 / (1 + np.exp(-spk_prediction))

        return waves, spk_prediction if self.enh_s2t_task else spk_prediction

    @torch.no_grad()
    def cal_permumation(self, ref_wavs, enh_wavs, criterion="si_snr"):
        """Calculate the permutation between seaprated streams in two adjacent segments.

        Args:
            ref_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            enh_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            criterion (str): one of ("si_snr", "mse", "corr)
        Returns:
            perm (torch.Tensor): permutation for enh_wavs (Batch, num_spk)
        """

        criterion_class = {"si_snr": SISNRLoss, "mse": FrequencyDomainMSE}[criterion]

        pit_solver = PITSolver(criterion=criterion_class())

        _, _, others = pit_solver(ref_wavs, enh_wavs)
        perm = others["perm"]
        return perm

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build DiarizeSpeech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            DiarizeSpeech: DiarizeSpeech instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return DiarizeSpeech(**kwargs)

    def permute_diar(self, waves, spk_prediction):
        # Permute the diarization result using the correlation
        # between wav and spk_prediction
        # FIXME(YushiUeda): batch_size > 1 is not considered
        num_spk = len(waves)
        permute_list = [np.array(p) for p in permutations(range(num_spk))]
        corr_list = []
        interp_prediction = F.interpolate(
            torch.sigmoid(spk_prediction).transpose(1, 2),
            size=waves[0].size(1),
            mode="linear",
        ).transpose(1, 2)
        for p in permute_list:
            diar_perm = interp_prediction[:, :, p]
            corr_perm = [0]
            for q in range(num_spk):
                corr_perm += np.corrcoef(
                    torch.squeeze(abs(waves[q])).cpu().numpy(),
                    torch.squeeze(diar_perm[:, :, q]).cpu().numpy(),
                )[0, 1]
            corr_list.append(corr_perm)
        max_corr, max_idx = torch.max(torch.from_numpy(np.array(corr_list)), dim=0)
        return (
            spk_prediction[:, :, permute_list[max_idx]],
            interp_prediction[:, :, permute_list[max_idx]],
            permute_list[max_idx],
        )

    def encode(self, speech, lengths):
        if self.enh_s2t_task:
            encoder_out, encoder_out_lens, _ = self.diar_model.encode_diar(
                speech, lengths, self.num_spk
            )
        else:
            bottleneck_feats = bottleneck_feats_lengths = None
            encoder_out, encoder_out_lens = self.diar_model.encode(
                speech, lengths, bottleneck_feats, bottleneck_feats_lengths
            )
        return encoder_out, encoder_out_lens

    def decode(self, encoder_out, encoder_out_lens):
        # SA-EEND
        if self.diar_model.attractor is None:
            assert self.num_spk is not None, 'Argument "num_spk" must be specified'
            spk_prediction = self.diar_model.decoder(encoder_out, encoder_out_lens)
            num_spk = self.num_spk
        # EEND-EDA
        else:
            # if num_spk is specified, use that number
            if self.num_spk is not None:
                attractor, att_prob = self.diar_model.attractor(
                    encoder_out,
                    encoder_out_lens,
                    to_device(
                        torch.zeros(
                            encoder_out.size(0),
                            self.num_spk + 1,
                            encoder_out.size(2),
                        ),
                        device=self.device,
                    ),
                )
                spk_prediction = torch.bmm(
                    encoder_out,
                    attractor[:, : self.num_spk, :].permute(0, 2, 1),
                )
                num_spk = self.num_spk
            # else find the first att_prob[i] < 0
            else:
                max_num_spk = 15  # upper bound number for estimation
                attractor, att_prob = self.diar_model.attractor(
                    encoder_out,
                    encoder_out_lens,
                    to_device(
                        torch.zeros(
                            encoder_out.size(0),
                            max_num_spk + 1,
                            encoder_out.size(2),
                        ),
                        device=self.device,
                    ),
                )
                att_prob = torch.squeeze(att_prob)
                for num_spk in range(len(att_prob)):
                    if att_prob[num_spk].item() < 0:
                        break
                spk_prediction = torch.bmm(
                    encoder_out, attractor[:, :num_spk, :].permute(0, 2, 1)
                )
        return spk_prediction, num_spk


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    allow_variable_data_keys: bool,
    segment_size: Optional[float],
    hop_size: Optional[float],
    normalize_segment_scale: bool,
    show_progressbar: bool,
    num_spk: Optional[int],
    normalize_output_wav: bool,
    multiply_diar_result: bool,
    enh_s2t_task: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build separate_speech
    diarize_speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        show_progressbar=show_progressbar,
        normalize_output_wav=normalize_output_wav,
        num_spk=num_spk,
        device=device,
        dtype=dtype,
        multiply_diar_result=multiply_diar_result,
        enh_s2t_task=enh_s2t_task,
    )
    diarize_speech = DiarizeSpeech.from_pretrained(
        model_tag=model_tag,
        **diarize_speech_kwargs,
    )

    # 3. Build data-iterator
    loader = DiarizationTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=DiarizationTask.build_preprocess_fn(
            diarize_speech.diar_train_args, False
        ),
        collate_fn=DiarizationTask.build_collate_fn(
            diarize_speech.diar_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    writer = NpyScpWriter(f"{output_dir}/predictions", f"{output_dir}/diarize.scp")

    if enh_s2t_task:
        wav_writers = []
        if diarize_speech.num_spk is not None:
            for i in range(diarize_speech.num_spk):
                wav_writers.append(
                    SoundScpWriter(
                        f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp"
                    )
                )
        else:
            for i in range(diarize_speech.diar_model.max_num_spk):
                wav_writers.append(
                    SoundScpWriter(
                        f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp"
                    )
                )

    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        if enh_s2t_task:
            waves, spk_predictions = diarize_speech(**batch)
            for b in range(batch_size):
                writer[keys[b]] = spk_predictions[b]
                for (spk, w) in enumerate(waves):
                    wav_writers[spk][keys[b]] = fs, w[b]
        else:
            spk_predictions = diarize_speech(**batch)
            for b in range(batch_size):
                writer[keys[b]] = spk_predictions[b]

    if enh_s2t_task:
        for w in wav_writers:
            w.close()
    writer.close()


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speaker Diarization inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--fs",
        type=humanfriendly_parse_size_or_none,
        default=8000,
        help="Sampling rate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Diarization training configuration",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Diarization model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )

    group = parser.add_argument_group("Data loading related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group = parser.add_argument_group("Diarize speech related")
    group.add_argument(
        "--segment_size",
        type=float,
        default=None,
        help="Segment length in seconds for segment-wise speaker diarization",
    )
    group.add_argument(
        "--hop_size",
        type=float,
        default=None,
        help="Hop length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--show_progressbar",
        type=str2bool,
        default=False,
        help="Whether to show a progress bar when performing segment-wise speaker "
        "diarization",
    )
    group.add_argument(
        "--num_spk",
        type=int_or_none,
        default=None,
        help="Predetermined number of speakers for inference",
    )

    group = parser.add_argument_group("Enh + Diar related")
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="enhancement and diarization joint model",
    )
    group.add_argument(
        "--normalize_segment_scale",
        type=str2bool,
        default=False,
        help="Whether to normalize the energy of the separated streams in each segment",
    )
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Whether to normalize the predicted wav to [-1~1]",
    )
    group.add_argument(
        "--multiply_diar_result",
        type=str2bool,
        default=False,
        help="Whether to multiply diar results to separated waves",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
