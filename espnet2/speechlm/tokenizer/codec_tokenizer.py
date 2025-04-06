#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from inspect import signature

import numpy as np
import torch
import yaml

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.speechlm.tokenizer.beats_tokenizer import (  # noqa
    BeatsRandomTokenizer,
    BeatsTokenizer,
    BeatsTokenizerConfig,
)
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


class CodecTokenizer(AbsTokenizer):
    """Codec Tokenizer implementation.

    Use cases:
        - use encode and decode for discrete (de)tokenization
        - use encode_continuous and decode_continuous for continuous
          (de)tokenization
        - use forward and detokenization for discrete (de)tokenization
          with flatten sequence style, which is more friendly for
          speechlm task
    """

    def __init__(
        self,
        codec_choice: str,
        codec_fs: int,
        device: str = "cpu",
        dump_audio: bool = False,
        checkpoint_path: str = None,
        config_path: str = None,
        max_token_per_frame: int = 32,
    ):
        """Codec Tokenizer initialization.

        Each of the codec implementation should assign all following features:
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
            from espnet2.tasks.gan_codec import GANCodecTask

            model, _ = GANCodecTask.build_model_from_file(
                config_path,
                checkpoint_path,
                device=str(device),
            )
            self.codec = model

            meta_info = self.codec.meta_info()
            self.n_codebook = min(meta_info["num_streams"], max_token_per_frame)
            self.size_codebook = meta_info["code_size_per_stream"][0]
            self.sample_rate = meta_info["fs"]
            self.subsample = meta_info["frame_shift"]

        elif self.codec_choice == "DAC":
            try:
                import dac
            except ImportError:
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
            except ImportError:
                raise ImportError("Please install Encodec with: pip install -U encodec")

            model_name = "encodec_model_" + str(codec_fs).replace("000", "khz")
            self.codec = getattr(EncodecModel, model_name)().to(device)
            # NOTE (Jinchuan): This Encodec model has 32 codebooks,
            # which is not necessary in usual cases.
            # We only adopt 8 first codebooks, a.k.a., 6kbps.
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
            except ImportError:
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

        elif self.codec_choice == "beats":
            beats_config = None
            if config_path:
                with open(config_path, "r") as f:
                    beats_config = yaml.safe_load(f)
            valid_args = signature(BeatsTokenizer.__init__).parameters
            remaining_args = (
                {k: v for k, v in beats_config.items() if k in valid_args}
                if beats_config
                else {}
            )
            self.codec = BeatsTokenizer(
                beats_tokenizer_ckpt_path=checkpoint_path,
                tokenizer_config=beats_config,
                **remaining_args,
            )
            self.codec = self.codec.to(device)
            self.codec.eval()
            self.n_codebook = 1
            self.size_codebook = self.codec.quantize.num_tokens
            self.sample_rate = 16000
            self.subsample = 320

        elif self.codec_choice == "beats_random":
            # Beats like patch-based frontend, with bestrq for quantization
            set_all_random_seed(42)
            beats_config = None
            if config_path:
                with open(config_path, "r") as f:
                    beats_config = yaml.safe_load(f)
            valid_args = signature(BeatsRandomTokenizer.__init__).parameters
            remaining_args = (
                {k: v for k, v in beats_config.items() if k in valid_args}
                if beats_config
                else {}
            )
            self.codec = BeatsRandomTokenizer(
                tokenizer_config=beats_config,
                **remaining_args,
            )
            self.codec = self.codec.to(device)
            self.codec.eval()
            self.n_codebook = 1
            self.size_codebook = self.codec.config.quant_n
            self.sample_rate = 16000
            self.subsample = 320
        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def encode(self, wavs):
        """Convert audio waveforms into codec codes.

        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        """
        assert wavs.dim() == 3 and wavs.size(1) == 1

        if self.codec_choice == "ESPnet":
            codes = self.codec.encode(wavs)
            codes = codes.permute(1, 2, 0)[:, :, : self.n_codebook]

        elif self.codec_choice == "DAC":
            codes = self.codec.encode(wavs)[1]
            codes = codes.transpose(1, 2)

        elif self.codec_choice == "EnCodec":
            encoded_frames = self.codec.encode(wavs)
            codes = encoded_frames[0][0].transpose(1, 2)

        elif self.codec_choice == "inhouse":
            codes = self.codec.encode(wavs).permute(1, 2, 0)

        elif self.codec_choice == "beats" or self.codec_choice == "beats_random":
            wav_in = wavs.squeeze(1)
            if wav_in.max() > 1.0 or wav_in.min() < -1.0:
                # Beats expects input in range [-1, 1]
                wav_in = wav_in.to(torch.float32)
                wav_in = wav_in / 2**15
            # Assume no padding, all wavs are full length
            assert wav_in.shape[0] == 1, "BeatsTokenizer only supports batch size 1"
            wav_len = torch.LongTensor([wav_in.size(1)] * wav_in.size(0)).to(
                wav_in.device
            )
            codes = self.codec.encode(xs_pad=wav_in, ilens=wav_len).unsqueeze(-1)

        else:
            raise NotImplementedError

        return codes

    def encode_continuous(self, wavs):
        """Convert audio waveforms into continuous codec encoding results.

        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            z (torch.Tensor): float tensor in shape [B, T, D]
        """

        if self.codec_choice == "ESPnet":
            z = self.codec.encode_continuous(wavs)
            z = z.transpose(1, 2)

        elif self.codec_choice == "DAC":
            z = self.codec.encode(wavs)[0]
            z = z.transpose(1, 2)

        else:
            raise NotImplementedError(
                f"Codec {self.codec_choice} does not support `encode_continuous`."
            )

        return z

    def decode(self, codes):
        """Recover the waveform from the codes.

        Input:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """
        if self.codec_choice == "ESPnet":
            codes = codes.permute(2, 0, 1)
            waveform = self.codec.decode(codes).squeeze(1)

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
            raise NotImplementedError(
                f"Codec {self.codec_choice} does not support `decode`."
            )

        return waveform

    def decode_continuous(self, z):
        """Recover the waveform from the continuous representations of codec.

        Input:
            z (torch.Tensor): Float tensor in shape [B, T, D], codec
              continuous representations
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """
        if self.codec_choice == "ESPnet":
            z = z.transpose(1, 2)
            waveform = self.codec.decode_continuous(z).squeeze(1)

        elif self.codec_choice == "DAC":
            z = z.transpose(1, 2)
            waveform = self.codec.decode(z).squeeze(1)

        else:
            raise NotImplementedError(
                f"Codec {self.codec_choice} does not support `decode_continuous`."
            )

        return waveform

    def forward(self, wavs):
        """Convert audio waveforms into flatten codec codes and resynthesis the audio.

        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T * n_codebook],
            resyn_audio (torch.Tensor): float tensor in shape [B, n_samples]
        """
        codes = self.encode(wavs)

        if self.dump_audio:
            resyn_audio = self.decode(codes)
        else:
            resyn_audio = None

        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)

        return codes, resyn_audio

    def detokenize(self, codes, n_codebook=None):
        """Convert flatten codec codes into resynthesis the audio.

        Input:
            codes (torch.Tensor): int tensor in shape [B, T * n_codebook],
                or [T * n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample],
                or [n_sample]
        """

        has_batch = codes.dim() == 2
        if not has_batch:
            codes = codes.unsqueeze(0)

        B, Tnq = codes.size()
        n_codebook = self.n_codebook if n_codebook is None else n_codebook
        assert Tnq % n_codebook == 0, (n_codebook, codes.size())
        codes = codes.view(B, Tnq // self.n_codebook, self.n_codebook)

        for l_idx in range(n_codebook):
            codes[:, :, l_idx] -= l_idx * self.size_codebook

        waveform = self.decode(codes)
        if not has_batch:
            waveform = waveform.squeeze(0)

        return waveform


if __name__ == "__main__":
    # a simple use case for batch processing
    device = "cuda:0"
    codec = CodecTokenizer(
        codec_choice="ESPnet",
        codec_fs=16000,
        device=device,
        dump_audio=True,
        checkpoint_path="espnet_codec/16khz_soundstream/train.total_count.best.pth",
        config_path="espnet_codec/16khz_soundstream/config.yaml",
    )

    import soundfile as sf

    waveform, sr = sf.read("1272-128104-0004.wav")
    waveform = (
        torch.from_numpy(waveform).view(1, 1, -1).to(device).float()
    )  # [B, C, n_sample]
    waveform = waveform.repeat(2, 1, 1)

    with torch.no_grad():
        # discrete
        codes = codec.encode(waveform)
        print("cdoes: ", codes.size())
        resyn_audio = codec.decode(codes)
        print("audio1", resyn_audio.size())
        resyn_audio = resyn_audio[0].cpu().numpy()
        sf.write("resyn1.wav", resyn_audio, sr)

        # continuous
        z = codec.encode_continuous(waveform)
        print("z: ", z.size())
        resyn_audio2 = codec.decode_continuous(z)
        print("audio2", resyn_audio2.size())
        resyn_audio2 = resyn_audio2[0].cpu().numpy()
        sf.write("resyn2.wav", resyn_audio2, sr)

        # high level API for speechlm
        flatten_codes, _ = codec(waveform)
        print("flatten_codes", flatten_codes.size())
        resyn_audio3 = codec.detokenize(flatten_codes)
        print("resyn", resyn_audio3.size())
        resyn_audio3 = resyn_audio3[0].cpu().numpy()
        sf.write("resyn3.wav", resyn_audio3, sr)
