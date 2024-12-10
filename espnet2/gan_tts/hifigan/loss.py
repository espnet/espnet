# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HiFiGAN-related loss modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank


class GeneratorAdversarialLoss(torch.nn.Module):
    """
    Generator adversarial loss module.

    This module computes the adversarial loss for the generator in a GAN
    setting. The loss can be calculated using either Mean Squared Error (MSE)
    or Hinge loss based on the specified configuration during initialization.

    Attributes:
        average_by_discriminators (bool): Whether to average the loss by
            the number of discriminators.
        criterion (callable): Loss function used for computing the adversarial
            loss, either MSE or Hinge.

    Args:
        average_by_discriminators (bool): Whether to average the loss by
            the number of discriminators.
        loss_type (str): Loss type, either "mse" or "hinge".

    Returns:
        Tensor: Generator adversarial loss value.

    Examples:
        >>> loss_fn = GeneratorAdversarialLoss(loss_type="mse")
        >>> outputs = [torch.tensor([0.5]), torch.tensor([0.8])]
        >>> loss = loss_fn(outputs)
        >>> print(loss)

    Raises:
        AssertionError: If an unsupported loss_type is provided.
    """

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize GeneratorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(
        self,
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculate generator adversarial loss.

        This method computes the adversarial loss for the generator based on the
        outputs received from the discriminator(s). The loss is calculated using
        either mean squared error (MSE) or hinge loss, depending on the specified
        loss type during initialization.

        Args:
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]):
                Discriminator outputs, which can be provided as a list of
                discriminator outputs, a list of lists of discriminator
                outputs, or a single tensor.

        Returns:
            Tensor: The calculated generator adversarial loss value.

        Examples:
            >>> loss_fn = GeneratorAdversarialLoss(loss_type="mse")
            >>> outputs = [torch.tensor([0.5]), torch.tensor([0.3])]
            >>> loss = loss_fn(outputs)
            >>> print(loss)  # Outputs the loss value
        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)

        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return -x.mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """
        Discriminator adversarial loss module.

    This module computes the adversarial loss for the discriminator in a GAN setup.
    It can handle both "mse" and "hinge" loss types and allows for averaging across
    multiple discriminators.

    Attributes:
        average_by_discriminators (bool): Whether to average the loss by the number
            of discriminators.
        loss_type (str): Loss type, either "mse" or "hinge".

    Args:
        average_by_discriminators (bool): Whether to average the loss by the number
            of discriminators.
        loss_type (str): Loss type, "mse" or "hinge".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Discriminator real loss value and
        discriminator fake loss value.

    Examples:
        >>> discriminator_loss = DiscriminatorAdversarialLoss(loss_type='hinge')
        >>> outputs_hat = [torch.tensor([0.9, 0.1]), torch.tensor([0.8, 0.2])]
        >>> outputs = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
        >>> real_loss, fake_loss = discriminator_loss(outputs_hat, outputs)

    Raises:
        AssertionError: If the provided loss_type is not "mse" or "hinge".
    """

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate discriminator adversarial loss.

        This method computes the adversarial loss for the discriminator by
        comparing the outputs from the generator and the ground truth outputs.
        The loss is computed separately for real and fake outputs based on the
        specified loss type (MSE or hinge).

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]):
                Discriminator outputs, list of discriminator outputs, or list
                of list of discriminator outputs calculated from the generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]):
                Discriminator outputs, list of discriminator outputs, or list
                of list of discriminator outputs calculated from the ground truth.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            discriminator real loss value and the discriminator fake loss value.

        Examples:
            >>> discriminator_loss = DiscriminatorAdversarialLoss()
            >>> real_outputs = [torch.tensor([[0.9], [0.8]])]
            >>> fake_outputs = [torch.tensor([[0.2], [0.1]])]
            >>> real_loss, fake_loss = discriminator_loss(fake_outputs, real_outputs)
            >>> print(real_loss, fake_loss)
        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class FeatureMatchLoss(torch.nn.Module):
    """
        Feature matching loss module.

    This module calculates the feature matching loss used in generative adversarial
    networks (GANs) to ensure that the generated outputs have similar features to
    the ground truth outputs. The loss can be averaged across layers and
    discriminators, and it can optionally include the final outputs of the
    discriminators for loss calculation.

    Attributes:
        average_by_layers (bool): Whether to average the loss by the number of layers.
        average_by_discriminators (bool): Whether to average the loss by the number
            of discriminators.
        include_final_outputs (bool): Whether to include the final output of each
            discriminator for loss calculation.

    Args:
        average_by_layers (bool): Whether to average the loss by the number of layers.
        average_by_discriminators (bool): Whether to average the loss by the number
            of discriminators.
        include_final_outputs (bool): Whether to include the final output of each
            discriminator for loss calculation.

    Returns:
        Tensor: Feature matching loss value.

    Examples:
        >>> feature_match_loss = FeatureMatchLoss()
        >>> feats_hat = [torch.randn(2, 80, 100), torch.randn(2, 80, 100)]
        >>> feats = [torch.randn(2, 80, 100), torch.randn(2, 80, 100)]
        >>> loss = feature_match_loss(feats_hat, feats)
        >>> print(loss)
    """

    def __init__(
        self,
        average_by_layers: bool = True,
        average_by_discriminators: bool = True,
        include_final_outputs: bool = False,
    ):
        """Initialize FeatureMatchLoss module.

        Args:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            include_final_outputs (bool): Whether to include the final output of
                each discriminator for loss calculation.

        """
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(
        self,
        feats_hat: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        feats: Union[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """
                Calculate generator adversarial loss.

        Args:
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs.

        Returns:
            Tensor: Generator adversarial loss value.
        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss


class MelSpectrogramLoss(torch.nn.Module):
    """
        Mel-spectrogram loss module.

    This module computes the loss between the generated and ground truth
    mel-spectrograms. It can operate in either L1 loss or MSE loss mode.

    Attributes:
        wav_to_mel (LogMelFbank): An instance of the LogMelFbank class used to
            convert waveforms to mel-spectrograms.

    Args:
        fs (int): Sampling rate. Defaults to 22050.
        n_fft (int): FFT points. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 256.
        win_length (Optional[int]): Window length. If None, defaults to
            win_length = n_fft.
        window (str): Window type. Defaults to "hann".
        n_mels (int): Number of Mel basis. Defaults to 80.
        fmin (Optional[int]): Minimum frequency for Mel. Defaults to 0.
        fmax (Optional[int]): Maximum frequency for Mel. If None, defaults to
            fs / 2.
        center (bool): Whether to use center window. Defaults to True.
        normalized (bool): Whether to use normalized one. Defaults to False.
        onesided (bool): Whether to use onesided one. Defaults to True.
        log_base (Optional[float]): Log base value. Defaults to 10.0.

    Examples:
        # Initialize the MelSpectrogramLoss
        loss_fn = MelSpectrogramLoss()

        # Calculate the loss
        y_hat = torch.randn(1, 1, 16000)  # Example generated waveform
        y = torch.randn(1, 1, 16000)      # Example groundtruth waveform
        loss = loss_fn(y_hat, y)

    Note:
        This loss can be used for training generative models in tasks like
        text-to-speech or audio synthesis.
    """

    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        log_base: Optional[float] = 10.0,
    ):
        """Initialize Mel-spectrogram loss.

        Args:
            fs (int): Sampling rate.
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency for Mel.
            fmax (Optional[int]): Maximum frequency for Mel.
            center (bool): Whether to use center window.
            normalized (bool): Whether to use normalized one.
            onesided (bool): Whether to use oneseded one.
            log_base (Optional[float]): Log base value.

        """
        super().__init__()
        self.wav_to_mel = LogMelFbank(
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            center=center,
            normalized=normalized,
            onesided=onesided,
            log_base=log_base,
        )

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        spec: Optional[torch.Tensor] = None,
        use_mse: bool = False,
    ) -> torch.Tensor:
        """
            Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).
            spec (Optional[Tensor]): Groundtruth linear amplitude spectrum tensor
                (B, T, n_fft // 2 + 1). If provided, use it instead of groundtruth
                waveform.
            use_mse (bool): Whether to use mse_loss instead of l1.

        Returns:
            Tensor: Mel-spectrogram loss value.

        Examples:
            >>> loss_fn = MelSpectrogramLoss()
            >>> y_hat = torch.randn(2, 1, 16000)  # Generated waveform
            >>> y = torch.randn(2, 1, 16000)      # Groundtruth waveform
            >>> loss = loss_fn(y_hat, y)
            >>> print(loss)

        Note:
            This loss can be used in training neural networks for tasks such as
            speech synthesis, where the objective is to minimize the difference
            between the generated audio and the target audio in the Mel-spectrogram
            domain.
        """
        mel_hat, _ = self.wav_to_mel(y_hat.squeeze(1))
        if spec is None:
            mel, _ = self.wav_to_mel(y.squeeze(1))
        else:
            mel, _ = self.wav_to_mel.logmel(spec)
        if use_mse:
            mel_loss = F.mse_loss(mel_hat, mel)
        else:
            mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss
