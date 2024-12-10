# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based neural codec ESPnet model."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch < 1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetGANCodecModel(AbsGANESPnetModel):
    """
    ESPnet model for GAN-based neural codec task.

    This class implements a GAN-based neural codec model for audio
    processing tasks. It utilizes a generator and discriminator
    architecture to encode and decode audio waveforms.

    Attributes:
        codec (AbsGANCodec): An instance of a codec which contains the
            generator and discriminator modules required for encoding and
            decoding audio.

    Args:
        codec (AbsGANCodec): An instance of a codec that must have
            'generator' and 'discriminator' attributes.

    Raises:
        AssertionError: If the provided codec does not have the required
            'generator' or 'discriminator' attributes.

    Examples:
        >>> from espnet2.gan_codec import MyGANCodec  # hypothetical import
        >>> model = ESPnetGANCodecModel(codec=MyGANCodec())
        >>> audio_tensor = torch.randn(1, 16000)  # Example audio tensor
        >>> loss_info = model.forward(audio_tensor)
        >>> encoded = model.encode(audio_tensor)
        >>> decoded = model.decode(encoded)

    Note:
        This model is designed for tasks involving GAN-based audio codec
        processing. It supports encoding and decoding with options for
        continuous representations.
    """

    @typechecked
    def __init__(
        self,
        codec: AbsGANCodec,
    ):
        """Initialize ESPnetGANCodecModel module."""
        super().__init__()
        self.codec = codec
        assert hasattr(
            codec, "generator"
        ), "generator module must be registered as codec.generator"
        assert hasattr(
            codec, "discriminator"
        ), "discriminator module must be registered as codec.discriminator"

    def meta_info(self) -> Dict[str, Any]:
        """
        Return meta information of the codec.

        This method retrieves and returns the meta information associated
        with the codec being used in the ESPnetGANCodecModel. The meta
        information typically includes details such as the codec's
        architecture, configuration, and other relevant attributes.

        Returns:
            Dict[str, Any]: A dictionary containing the meta information
            of the codec.

        Examples:
            >>> model = ESPnetGANCodecModel(codec=my_codec)
            >>> info = model.meta_info()
            >>> print(info)
            {'architecture': 'GAN', 'version': '1.0', ...}

        Note:
            Ensure that the codec has a properly defined `meta_info`
            method to retrieve the necessary information.
        """
        return self.codec.meta_info()

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
            Return generator or discriminator loss in a dictionary format.

        This method processes the input audio through the GAN codec model and
        returns the computed loss along with various statistics. Depending on
        the `forward_generator` flag, it can either compute the generator's
        loss or the discriminator's loss.

        Args:
            audio (Tensor): Audio waveform tensor of shape (B, T_wav), where
                B is the batch size and T_wav is the number of audio samples.
            forward_generator (bool): Flag indicating whether to forward the
                generator. If True, the generator's loss is computed; if False,
                the discriminator's loss is computed.
            kwargs: Additional keyword arguments. The "utt_id" should be among
                the input if required.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor indicating the computed loss.
                - stats (Dict[str, float]): Dictionary of statistics to be
                  monitored during training.
                - weight (Tensor): Weight tensor used to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D)
                  indicating which model's parameters should be updated.

        Examples:
            >>> model = ESPnetGANCodecModel(codec)
            >>> audio_input = torch.randn(8, 16000)  # Example input tensor
            >>> result = model.forward(audio_input, forward_generator=True)
            >>> print(result['loss'])  # Accessing the computed loss

        Note:
            Ensure that the `codec` object passed during model initialization has
            the required `generator` and `discriminator` attributes.

        Raises:
            AssertionError: If the `codec` does not have the required attributes
            or if any input is invalid.
        """
        # Make the batch for codec inputs
        batch = dict(
            audio=audio,
            forward_generator=forward_generator,
        )

        return self.codec(**batch)

    def encode(self, audio: torch.Tensor, **kwargs):
        """
        Codec Encoding Process.

        This method encodes audio waveforms into codec representations. It
        handles different input tensor shapes, ensuring they are compatible
        with the encoding process. The resulting encoded output can be used
        for various applications in the GAN-based neural codec framework.

        Args:
            audio (Tensor): Audio waveform tensor, which can have the shape:
                - (B, 1, T_wav) for batched audio with a single channel.
                - (B, T_wav) for batched audio without explicit channel.
                - (T_wav) for a single audio sample.

        Returns:
            Tensor: Generated codecs in the shape (N_stream, B, T), where:
                - N_stream: Number of codec streams.
                - B: Batch size.
                - T: Length of the encoded representation.

        Examples:
            >>> model = ESPnetGANCodecModel(codec)
            >>> audio = torch.randn(2, 1, 16000)  # Example audio tensor
            >>> encoded = model.encode(audio)
            >>> print(encoded.shape)  # Output shape should be (N_stream, 2, T)

        Note:
            Ensure that the input audio tensor is properly shaped as
            described in the Args section for successful encoding.
        """

        # convert to [B, n_channle=1, n_sample] anyway
        if audio.dim() == 1:
            audio = audio.view(1, 1, -1)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        batch = dict(x=audio)
        batch.update(kwargs)
        return self.codec.encode(**batch)

    def encode_continuous(self, audio):
        """
        Codec Encoding Process without quantization.

        This method encodes the given audio input into a continuous codec
        representation without applying any quantization. It ensures that
        the audio input is reshaped appropriately before passing it to the
        encoder.

        Args:
            audio (Tensor): Audio waveform tensor with shapes:
                (B, 1, T_wav), (B, T_wav), or (T_wav).

        Returns:
            Tensor: Generated codes with shape (B, D, T), where B is the
            batch size, D is the dimension of the codes, and T is the
            temporal dimension.

        Examples:
            >>> model = ESPnetGANCodecModel(codec)
            >>> audio_input = torch.randn(2, 16000)  # Simulated audio
            >>> codes = model.encode_continuous(audio_input)
            >>> print(codes.shape)  # Output shape will be (2, D, T)

        Note:
            The method will automatically reshape the input tensor to
            ensure it has the appropriate dimensions for encoding.
        """

        # convert to [B, n_channle=1, n_sample] anyway
        if audio.dim() == 1:
            audio = audio.view(1, 1, -1)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return self.codec.generator.encoder(audio)

    def decode(
        self,
        codes: torch.Tensor,
    ):
        """
        Codec Decoding Process.

        This method decodes the provided codec tokens into an audio waveform.

        Args:
            codes (Tensor): codec tokens with shape [N_stream, B, T], where:
                - N_stream: Number of streams in the codec.
                - B: Batch size.
                - T: Length of the tokens.

        Returns:
            Tensor: Generated waveform with shape (B, 1, n_sample), where:
                - B: Batch size.
                - 1: Single channel for the audio waveform.
                - n_sample: Number of samples in the generated waveform.

        Examples:
            >>> codec_model = ESPnetGANCodecModel(codec)
            >>> codes = torch.randn(2, 4, 256)  # Example codec tokens
            >>> waveform = codec_model.decode(codes)
            >>> print(waveform.shape)  # Output shape: (4, 1, n_sample)

        Note:
            Ensure that the input `codes` tensor is in the correct format to
            avoid runtime errors.
        """
        return self.codec.decode(codes)

    def decode_continuous(
        self,
        z: torch.Tensor,
    ):
        """
            Codec Decoding Process without dequantization.

        This method takes a continuous codec representation and decodes it into
        an audio waveform. The input tensor should represent the continuous
        features obtained from the encoding process, and the output is a
        reconstructed waveform tensor.

        Args:
            z (Tensor): Continuous codec representation with shape (B, D, T),
                        where B is the batch size, D is the number of
                        dimensions, and T is the length of the sequence.

        Returns:
            Tensor: Generated waveform with shape (B, 1, n_sample), where n_sample
                    is the number of samples in the reconstructed waveform.

        Examples:
            >>> model = ESPnetGANCodecModel(codec)
            >>> continuous_codes = torch.randn(2, 256, 100)  # Example input
            >>> waveform = model.decode_continuous(continuous_codes)
            >>> print(waveform.shape)  # Output shape should be (2, 1, n_sample)

        Note:
            Ensure that the input tensor `z` is properly shaped as specified
            above to avoid runtime errors during the decoding process.
        """
        return self.codec.generator.decoder(z)

    def collect_feats(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate features and return them as a dictionary.

        This method processes the input audio waveform tensor and extracts
        relevant features, returning them in a dictionary format. The
        dictionary keys correspond to different types of features that can
        be used for further processing or analysis.

        Args:
            audio (Tensor): Audio waveform tensor of shape (B, T_wav),
                where B is the batch size and T_wav is the number of
                audio samples.
            kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Dict[str, Tensor]: A dictionary containing the extracted features,
                where each key is a string representing the feature name
                and the corresponding value is a tensor of the feature data.

        Examples:
            >>> model = ESPnetGANCodecModel(codec)
            >>> audio_input = torch.randn(2, 16000)  # Example audio tensor
            >>> features = model.collect_feats(audio_input)
            >>> print(features.keys())  # Output might include feature names

        Note:
            This method is currently a placeholder and returns an empty
            dictionary. Implement feature extraction logic as needed.
        """

        return {}
