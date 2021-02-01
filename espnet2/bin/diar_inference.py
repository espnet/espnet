#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from tqdm import trange
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.diar import DiarizationTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
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
		diar_train_config: Union[Path, str],
		diar_model_file: Union[Path, str] = None,
		segment_size: Optional[float] = None,
		hop_size: Optional[float] = None,
		normalize_segment_scale: bool = False,
		show_progressbar: bool = False,
		device: str = "cpu",
		dtype: str = "float32",
	):
	    assert check_argument_types()

	    # 1. Build Diar model
	    diar_model, diar_train_args = DiarizationTask.build_model_from_file(
	    	diar_train_config, diar_model_file, device
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

        self.num_spk = enh_model.num_spk

        self.segmenting = segment_size is not None and hop_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speech %s" % task)
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speech %s on the input" % task)


    @torch.no_grad()
    def __call__(
        self, speech_mix: Union[torch.Tensor, np.ndarray], fs: int = 8000
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [separated_audio1, separated_audio2, ...]

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        # lenghts: (B,)
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        # a. To device
        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if self.segmenting and lengths[0] > self.segment_size * fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]  # B x T [x C]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Enhancement/Separation Forward
                feats, f_lens = self.enh_model.encoder(speech_seg, lengths_seg)
                feats, _, _ = self.enh_model.separator(feats, f_lens)
                processed_wav = [
                    self.enh_model.decoder(f, lengths_seg)[0] for f in feats
                ]
                if speech_seg.dim() > 2:
                    # multi-channel speech
                    speech_seg_ = speech_seg[:, self.ref_channel]
                else:
                    speech_seg_ = speech_seg

                if self.normalize_segment_scale:
                    # normalize the energy of each separated stream
                    # to match the input energy
                    processed_wav = [
                        self.normalize_scale(w, speech_seg_) for w in processed_wav
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
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            # ensure the stitched length is same as input
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            # b. Enhancement/Separation Forward
            feats, f_lens = self.enh_model.encoder(speech_mix, lengths)
            feats, _, _ = self.enh_model.separator(feats, f_lens)
            waves = [self.enh_model.decoder(f, lengths)[0] for f in feats]

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                for w in waves
            ]  # list[(batch, sample)]
        else:
            waves = [w.cpu().numpy() for w in waves]

        return waves

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
        loss_func = {
            "si_snr": self.enh_model.si_snr_loss,
            "mse": lambda enh, ref: torch.mean((enh - ref).pow(2), dim=1),
            "corr": lambda enh, ref: (
                (enh * ref).sum(dim=1)
                / (enh.pow(2).sum(dim=1) * ref.pow(2).sum(dim=1) + EPS)
            ).clamp(min=EPS, max=1 - EPS),
        }[criterion]

        _, perm = self.enh_model._permutation_loss(ref_wavs, enh_wavs, loss_func)
        return perm


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)