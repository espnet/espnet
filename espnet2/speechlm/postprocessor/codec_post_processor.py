from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch

from espnet2.speechlm.postprocessor.abs_postprocessor import AbsPostProcessor


# Used in both data preparation stage and inference stage.
class Codec_Tokenizer(torch.nn.Module):
    def __init__(self, codec_choice, codec_fs, device, dump_audio=False):
        super(Codec_Tokenizer, self).__init__()
        self.codec_choice = codec_choice
        self.device = device
        self.dump_audio = dump_audio

        if self.codec_choice == "DAC":
            try:
                import dac
            except:
                raise ImportError(
                    "Please install DAC with: pip install descript-audio-codec"
                )

            model_path = dac.utils.download(
                model_type=str(codec_fs).replace("000", "khz")
            )
            self.codec = dac.DAC.load(model_path).to(device)
            self.n_codebook = self.codec.n_codebooks
            self.size_codebook = self.codec.codebook_size
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder_rates)

        elif self.codec_choice == "EnCodec":
            try:
                from encodec import EncodecModel
            except:
                raise ImportError("Please install Encodec with: pip install -U encodec")

            model_name = "encodec_model_" + str(codec_fs).replace("000", "khz")
            self.codec = getattr(EncodecModel, model_name)().to(device)
            bandwidth = 6.0  # 8 codebooks
            # bandwidth = max(self.codec.target_bandwidths) # 32 codebooks, too large
            self.codec.set_target_bandwidth(bandwidth)
            self.n_codebook = self.codec.quantizer.get_num_quantizers_for_bandwidth(
                self.codec.frame_rate, bandwidth
            )
            self.size_codebook = self.codec.quantizer.bins
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder.ratios)

        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def decode(self, codes):
        if self.codec_choice == "DAC":
            raise NotImplementedError
        elif self.codec_choice == "EnCodec":
            encoded_frames = [(codes.transpose(1, 2), None)]
            waveform = self.codec.decode(encoded_frames)
        else:
            raise NotImplementedError

        return waveform

    def __call__(self, wavs):
        # All wavs in shape of [batch_size, 1, num_samples]
        assert wavs.dim() == 3 and wavs.size(1) == 1

        if self.codec_choice == "DAC":
            z, codes = self.codec.encode(wavs)[:2]
            codes = codes.transpose(1, 2)
            if self.dump_audio:
                raise NotImplementedError
            else:
                resyn_audio = None

        elif self.codec_choice == "EnCodec":
            encoded_frames = self.codec.encode(wavs)
            codes = encoded_frames[0][0].transpose(1, 2)
            if self.dump_audio:
                resyn_audio = self.codec.decode(encoded_frames).squeeze(1)
            else:
                resyn_audio = None

        else:
            raise NotImplementedError

        # All codes in shape of [batch_size, T, n_codebook]
        # All resyn_audio in shape of [batch_size, num_samples]
        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)

        return codes, resyn_audio


class CodecPostProcessor(AbsPostProcessor):
    """The abstract Post-Processor class for SpeechLM"""

    def __init__(
        self,
        codec_choice: str = "encodec",
        codec_fs: int = 24000,
        device: str = "cpu",
        dump_audio: bool = True,
    ):
        super(CodecPostProcessor, self).__init__()

        self.tokenizer = Codec_Tokenizer(
            codec_choice=codec_choice,
            codec_fs=codec_fs,
            device=device,
            dump_audio=dump_audio,
        )
        self.sample_rate = self.tokenizer.sample_rate

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        print(tokens.device)

        for t in range(tokens.size(1)):
            tokens[:, t] -= t * self.tokenizer.size_codebook

        waveform = self.tokenizer.decode(tokens.unsqueeze(0)).squeeze(0)
        return waveform
