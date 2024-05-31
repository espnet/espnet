#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer


class CodecTokenizer(AbsTokenizer):
    def __init__(
        self,
        codec_choice: str,
        codec_fs: int,
        device: str,
        dump_audio: bool = False,
        checkpoint_path: str = None,
        config_path: str = None,
        max_token_per_frame: int = 8,
    ):
        """Codec Tokenizer implementation that is used in:
        (1) waveform tokenization during data prep stage;
        (2) audio token detokenization after SpeechLM inference stage.

        It contains multiple audio codec implementations, controlled by
        the "codec_choice" argument. For any codec model, this implementation
        object contains the following attributes:
            self.n_codebook (int): the number of codec codebooks.
            self.size_codebook (int): the dimension of codebooks.
            self.sample_rate (int): the sample rate the model trained on.
            self.subsample (int): the subsample rate, a.k.a., frame shift.
        """
        super(CodecTokenizer, self).__init__()
        self.codec_choice = codec_choice
        self.device = device
        self.dump_audio = dump_audio

        if self.codec_choice == "ESPnet":
            from espnet2.bin.gan_codec_inference import AudioCoding

            model = AudioCoding(
                train_config=config_path,
                model_file=checkpoint_path,
                device=str(device),
            )
            self.codec = model

            meta_info = self.codec.model.meta_info()
            self.n_codebook = min(meta_info["num_streams"], max_token_per_frame)
            self.size_codebook = meta_info["code_size_per_stream"][0]
            self.sample_rate = meta_info["fs"]
            self.subsample = meta_info["frame_shift"]

        elif self.codec_choice == "DAC":
            try:
                import dac
            except:
                raise ImportError("Install DAC with: pip install descript-audio-codec")

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
            # NOTE (Jinchuan): This Encodec model has 32 codebooks, which is not necessary
            # in usual cases. We only adopt 8 first codebooks, a.k.a., 6kbps.
            bandwidth = 6.0
            self.codec.set_target_bandwidth(bandwidth)
            self.n_codebook = self.codec.quantizer.get_num_quantizers_for_bandwidth(
                self.codec.frame_rate, bandwidth
            )
            self.size_codebook = self.codec.quantizer.bins
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder.ratios)
        
        elif self.codec_choice == "inhouse":
            try:
                from models.soundstream import SoundStream
                from omegaconf import OmegaConf
            except:
                raise ImportError("fail to use inhouse codec")

            model_path = "encodec_16k_6kbps_multiDisc/ckpt_01135000.pth"
            model_config = "encodec_16k_6kbps_multiDisc/config.yaml"
            config = OmegaConf.load(model_config)
            model = SoundStream(**config.generator.config)

            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict["codec_model"])
            model = model.to(device)
            self.codec = model

            self.n_codebook = 8
            self.sample_rate = 16000
            self.size_codebook = 1024
            self.subsample = 320

        else:
            raise ValueError(f"Codec {codec_choice} is not supported")


    @torch.no_grad()
    def decode(self, codes):
        """
        Recover the waveform from the codes.
        Input:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """
        if self.codec_choice == "ESPnet":
            codes = codes.permute(2, 0, 1)
            waveform = self.codec.decode(codes)["resyn_audio"].squeeze(1)

        elif self.codec_choice == "DAC":
            z = self.codec.quantizer.from_codes(codes.transpose(1, 2))[0]
            waveform = self.codec.decode(z).squeeze(1)

        elif self.codec_choice == "EnCodec":
            encoded_frames = [(codes.transpose(1, 2), None)]
            waveform = self.codec.decode(encoded_frames).squeeze(1)

        elif self.codec_choice == "inhouse":
            codes = codes.permute(2, 0, 1)
            wav = self.codec.decode(codes).squeeze(1)
            return wav

        else:
            raise NotImplementedError

        return waveform

    @torch.no_grad()
    def forward(self, wavs):
        """
        Convert audio waveforms into codec codes
        Input:
            wavs (torch.Tensor): float tensor in shape [B, n_channel, n_sample],
                currently only n_channel=1 is supported.
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
            resyn_audio (torch.Tensor): float tensor in shape [B, n_sample],
                if self.dump_audio else None, the resynthesized audio based on the
                codec codes.
        """
        assert wavs.dim() == 3 and wavs.size(1) == 1

        # (1) Tokenization
        # All codes in shape of [batch_size, T, n_codebook]
        if self.codec_choice == "ESPnet":

            # TODO(Jinchuan): pin jiatong to support batch inference
            assert wavs.size(0) == 1, "ESPnet codec doesn't support batch inference"
            codes = self.codec(wavs.view(-1), encode_only=False)["codes"]
            codes = codes.permute(1, 2, 0)[:, :, : self.n_codebook]

        elif self.codec_choice == "DAC":
            z, codes = self.codec.encode(wavs)[:2]
            codes = codes.transpose(1, 2)

        elif self.codec_choice == "EnCodec":
            encoded_frames = self.codec.encode(wavs)
            codes = encoded_frames[0][0].transpose(1, 2)
        
        elif self.codec_choice == "inhouse":
            codes = self.codec.encode(wavs).permute(1, 2, 0)

        else:
            raise NotImplementedError

        # (2) Detokenization
        # All resyn_audio in shape of [batch_size, n_sample]
        if self.dump_audio:
            resyn_audio = self.decode(codes)
        else:
            resyn_audio = None

        # (3) shift by codebook
        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)

        return codes, resyn_audio
