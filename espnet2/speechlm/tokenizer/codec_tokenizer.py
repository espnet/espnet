#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer


class CodecTokenizer(AbsTokenizer):
    """
        CodecTokenizer is a tokenizer implementation for various audio codecs.

    This class provides methods for encoding and decoding audio waveforms using
    different codec implementations. It supports both discrete and continuous
    tokenization, allowing for flexible audio processing in speech language
    models.

    Use cases:
        - Use `encode` and `decode` for discrete (de)tokenization.
        - Use `encode_continuous` and `decode_continuous` for continuous
          (de)tokenization.
        - Use `forward` and `detokenize` for discrete (de)tokenization with
          flatten sequence style, which is more friendly for speechlm tasks.

    Attributes:
        codec_choice (str): The chosen codec implementation.
        device (str): The device for model computation (e.g., "cpu" or "cuda").
        dump_audio (bool): Flag to indicate whether to dump the audio during
            processing.
        n_codebook (int): The number of codec codebooks.
        size_codebook (int): The dimension of codebooks.
        sample_rate (int): The sample rate the model was trained on.
        subsample (int): The subsample rate, a.k.a., frame shift.

    Args:
        codec_choice (str): The codec implementation to use. Options include
            "ESPnet", "DAC", "EnCodec", and "inhouse".
        codec_fs (int): The sample rate for the codec.
        device (str, optional): The device to run the model on. Defaults to "cpu".
        dump_audio (bool, optional): Whether to dump the audio during processing.
            Defaults to False.
        checkpoint_path (str, optional): Path to the model checkpoint file.
            Defaults to None.
        config_path (str, optional): Path to the model configuration file.
            Defaults to None.
        max_token_per_frame (int, optional): Maximum number of tokens per frame.
            Defaults to 32.

    Raises:
        ValueError: If an unsupported codec choice is provided.
        ImportError: If the required codec library is not installed.

    Examples:
        To initialize the CodecTokenizer and encode/decode audio waveforms:

        ```python
        device = "cuda:0"
        codec = CodecTokenizer(
            codec_choice="ESPnet",
            codec_fs=16000,
            device=device,
            dump_audio=True,
            checkpoint_path="path/to/checkpoint.pth",
            config_path="path/to/config.yaml",
        )

        # Encode audio
        waveform = torch.randn(1, 1, 16000)  # Example waveform
        codes = codec.encode(waveform)

        # Decode audio
        reconstructed_waveform = codec.decode(codes)
        ```

    Note:
        The `encode` and `decode` methods are designed to work with audio tensors
        in specific shapes. Ensure the input tensors are formatted correctly.

    Todo:
        - Add support for additional codecs as needed.
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
        """Codec Tokenizer initialization

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

        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def encode(self, wavs):
        """
        Convert audio waveforms into codec codes.

        Args:
            wavs (torch.Tensor): A float tensor of shape [B, 1, n_sample],
                where B is the batch size and n_sample is the number of audio
                samples.

        Returns:
            torch.Tensor: An integer tensor of shape [B, T, n_codebook],
                where T is the number of time frames produced by the encoding.

        Raises:
            AssertionError: If the input tensor does not have 3 dimensions or
                if the second dimension is not equal to 1.

        Examples:
            >>> import torch
            >>> codec = CodecTokenizer(codec_choice="ESPnet", codec_fs=16000)
            >>> wavs = torch.randn(2, 1, 32000)  # Example batch of audio
            >>> codes = codec.encode(wavs)
            >>> print(codes.shape)
            torch.Size([2, T, n_codebook])
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

        else:
            raise NotImplementedError

        return codes

    def encode_continuous(self, wavs):
        """
        Convert audio waveforms into continuous codec encoding results.

        This method processes the input audio waveforms and converts them into
        continuous codec representations. The shape of the input tensor should
        be [B, 1, n_sample], where B is the batch size, and n_sample is the
        number of samples in the audio waveform. The output tensor will have
        the shape [B, T, D], where T is the number of time frames and D is
        the dimensionality of the continuous representation.

        Args:
            wavs (torch.Tensor): A float tensor of shape [B, 1, n_sample]
                representing the audio waveforms to be encoded.

        Returns:
            torch.Tensor: A float tensor of shape [B, T, D] containing the
            continuous codec encoding results.

        Raises:
            NotImplementedError: If the codec choice is not supported.

        Examples:
            >>> import torch
            >>> codec = CodecTokenizer(codec_choice="ESPnet", codec_fs=16000)
            >>> wavs = torch.randn(2, 1, 32000)  # Example input
            >>> continuous_encoding = codec.encode_continuous(wavs)
            >>> print(continuous_encoding.shape)
            torch.Size([2, T, D])  # T and D depend on the codec implementation
        """

        if self.codec_choice == "ESPnet":
            z = self.codec.encode_continuous(wavs)
            z = z.transpose(1, 2)

        elif self.codec_choice == "DAC":
            z = self.codec.encode(wavs)[0]
            z = z.transpose(1, 2)

        else:
            raise NotImplementedError

        return z

    def decode(self, codes):
        """
        Recover the waveform from the codes.

        Args:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook].

        Returns:
            waveform (torch.Tensor): float tensor in shape [B, n_sample].

        Raises:
            NotImplementedError: If the codec_choice is not supported.

        Examples:
            >>> tokenizer = CodecTokenizer(codec_choice="ESPnet", codec_fs=16000)
            >>> codes = torch.randint(0, 256, (2, 10, 8))  # Example codes
            >>> waveform = tokenizer.decode(codes)
            >>> print(waveform.shape)  # Output shape: [2, n_sample]
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
            raise NotImplementedError

        return waveform

    def decode_continuous(self, z):
        """
                Recover the waveform from the continuous representations of codec.

        This method takes continuous representations (also known as latent
        variables) produced by the codec and reconstructs the audio waveform.
        It is particularly useful for processing audio data in a continuous
        form rather than discrete tokens.

        Args:
            z (torch.Tensor): Float tensor in shape [B, T, D], where B is the
            batch size, T is the number of time steps, and D is the dimension
            of the codec continuous representations.

        Returns:
            waveform (torch.Tensor): Float tensor in shape [B, n_sample],
            representing the reconstructed audio waveform.

        Raises:
            NotImplementedError: If the codec choice is not supported.

        Examples:
            >>> # Assuming 'codec' is an instance of CodecTokenizer
            >>> z = torch.randn(2, 100, 512)  # Example continuous representations
            >>> waveform = codec.decode_continuous(z)
            >>> print(waveform.shape)
            torch.Size([2, n_sample])  # n_sample will depend on the codec used
        """
        if self.codec_choice == "ESPnet":
            z = z.transpose(1, 2)
            waveform = self.codec.decode_continuous(z).squeeze(1)

        elif self.codec_choice == "DAC":
            z = z.transpose(1, 2)
            waveform = self.codec.decode(z).squeeze(1)

        else:
            raise NotImplementedError

        return waveform

    def forward(self, wavs):
        """
        Convert audio waveforms into flatten codec codes and resynthesize the audio.

        This method processes input audio waveforms to generate a flattened
        representation of codec codes while optionally resynthesizing the audio.
        It combines encoding and decoding in a single step, which is particularly
        useful for speech-related tasks.

        Args:
            wavs (torch.Tensor): Float tensor in shape [B, 1, n_sample], where B
                is the batch size and n_sample is the number of audio samples.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - codes (torch.Tensor): Int tensor in shape [B, T * n_codebook],
                  representing the flattened codec codes.
                - resyn_audio (torch.Tensor or None): Float tensor in shape
                  [B, n_samples] if `self.dump_audio` is True, containing the
                  resynthesized audio waveforms; otherwise, it returns None.

        Examples:
            >>> codec = CodecTokenizer(codec_choice="ESPnet", codec_fs=16000)
            >>> wavs = torch.randn(2, 1, 16000)  # Example input tensor
            >>> codes, resyn_audio = codec.forward(wavs)
            >>> print(codes.shape)  # Should print shape [2, T * n_codebook]
            >>> print(resyn_audio.shape)  # Shape depends on the decoding

        Note:
            The method modifies the input codes by adding a shift based on the
            number of codebooks and their sizes before flattening them.
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
        """
        Convert flatten codec codes into resynthesis the audio.

        Args:
            codes (torch.Tensor): int tensor in shape [B, T * n_codebook],
                or [T * n_codebook]. The flattened codec codes to be
                converted back into audio.
            n_codebook (int, optional): The number of codebooks used for
                encoding. If not provided, the default number of codebooks
                from the instance will be used.

        Returns:
            waveform (torch.Tensor): float tensor in shape [B, n_sample],
                or [n_sample]. The resynthesized audio waveform from the
                provided codec codes.

        Raises:
            AssertionError: If the total number of tokens is not divisible
            by the number of codebooks.

        Examples:
            >>> codec = CodecTokenizer(codec_choice="ESPnet", codec_fs=16000)
            >>> flatten_codes = torch.randint(0, 256, (1, 32))  # Example codes
            >>> audio_waveform = codec.detokenize(flatten_codes)
            >>> print(audio_waveform.shape)
            torch.Size([1, n_sample])
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
        print(f"cdoes: ", codes.size())
        resyn_audio = codec.decode(codes)
        print(f"audio1", resyn_audio.size())
        resyn_audio = resyn_audio[0].cpu().numpy()
        sf.write("resyn1.wav", resyn_audio, sr)

        # continuous
        z = codec.encode_continuous(waveform)
        print(f"z: ", z.size())
        resyn_audio2 = codec.decode_continuous(z)
        print(f"audio2", resyn_audio2.size())
        resyn_audio2 = resyn_audio2[0].cpu().numpy()
        sf.write("resyn2.wav", resyn_audio2, sr)

        # high level API for speechlm
        flatten_codes, _ = codec(waveform)
        print(f"flatten_codes", flatten_codes.size())
        resyn_audio3 = codec.detokenize(flatten_codes)
        print("resyn", resyn_audio3.size())
        resyn_audio3 = resyn_audio3[0].cpu().numpy()
        sf.write("resyn3.wav", resyn_audio3, sr)
