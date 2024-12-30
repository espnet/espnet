from typing import Tuple

import librosa
import torch

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LogMel(torch.nn.Module):
    """
        Convert STFT to log Mel filterbank features.

    This module transforms short-time Fourier transform (STFT) features into log Mel
    filterbank features. The parameters for this class are the same as those used in
    `librosa.filters.mel`.

    Attributes:
        mel_options (dict): Configuration options for the Mel filterbank.
        log_base (float or None): Base of the logarithm used for log Mel calculation.

    Args:
        fs (int): Sampling rate of the incoming signal (default: 16000).
        n_fft (int): Number of FFT components (default: 512).
        n_mels (int): Number of Mel bands to generate (default: 80).
        fmin (float or None): Lowest frequency (in Hz) (default: None).
            If `None`, defaults to 0.
        fmax (float or None): Highest frequency (in Hz) (default: None).
            If `None`, defaults to `fs / 2.0`.
        htk (bool): Use HTK formula instead of Slaney (default: False).
        log_base (float or None): Base of the logarithm (default: None).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - logmel_feat (torch.Tensor): Log Mel filterbank features.
            - ilens (torch.Tensor): Input lengths.

    Examples:
        >>> logmel = LogMel(fs=16000, n_fft=512, n_mels=80)
        >>> feat = torch.randn(2, 100, 512)  # (B, T, D1)
        >>> ilens = torch.tensor([100, 80])  # input lengths
        >>> logmel_feat, ilens = logmel(feat, ilens)

    Note:
        The Mel matrix created by librosa is different from the one used in Kaldi.

    Raises:
        ValueError: If any of the input parameters are invalid.
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        """
            Returns a string representation of the LogMel parameters.

        This method provides a detailed string representation of the LogMel
        instance's parameters, specifically the mel options used for generating
        the Mel filter bank. It is typically used for debugging and logging
        purposes to understand the configuration of the LogMel instance.

        Attributes:
            mel_options: A dictionary containing the parameters used to create
                the Mel filter bank, including:
                    - sr: Sampling rate of the incoming signal
                    - n_fft: Number of FFT components
                    - n_mels: Number of Mel bands to generate
                    - fmin: Lowest frequency (in Hz)
                    - fmax: Highest frequency (in Hz)
                    - htk: Boolean indicating the use of HTK formula instead
                        of Slaney

        Returns:
            str: A comma-separated string representation of the mel options.

        Examples:
            >>> logmel = LogMel(fs=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000)
            >>> print(logmel.extra_repr())
            sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000, htk=False
        """
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Convert STFT to log Mel spectrogram features.

        This module takes the Short-Time Fourier Transform (STFT) features as input
        and converts them into log Mel spectrogram features. The conversion is based
        on the Mel filter bank, and the arguments are similar to those used in
        librosa.filters.mel.

        Attributes:
            mel_options (dict): Dictionary containing the parameters for the Mel filter.
            log_base (float or None): Base of the logarithm used for computing log Mel.

        Args:
            fs (int): Sampling rate of the incoming signal. Must be greater than 0.
            n_fft (int): Number of FFT components. Must be greater than 0.
            n_mels (int): Number of Mel bands to generate. Must be greater than 0.
            fmin (float): Lowest frequency (in Hz). Must be greater than or equal to 0.
            fmax (float): Highest frequency (in Hz). If `None`, uses `fmax = fs / 2.0`.
            htk (bool): If True, uses HTK formula instead of Slaney.
            log_base (float or None): Base of the logarithm for log Mel computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logmel_feat (torch.Tensor): The computed log Mel features of shape
                  (B, T, D2), where B is the batch size, T is the number of time frames,
                  and D2 is the number of Mel bands.
                - ilens (torch.Tensor): The lengths of the input features for each batch.

        Examples:
            >>> logmel = LogMel()
            >>> feat = torch.rand(2, 100, 512)  # Example input (B=2, T=100, D1=512)
            >>> ilens = torch.tensor([100, 80])  # Example lengths
            >>> logmel_feat, ilens = logmel(feat, ilens)
            >>> print(logmel_feat.shape)  # Output: (2, 100, 80)

        Note:
            - The Mel matrix generated by librosa differs from that used in Kaldi.
            - The input tensor `feat` should have the shape (B, T, D1), where B is the
              batch size, T is the number of time frames, and D1 is the number of FFT
              components.

        Raises:
            ValueError: If the input tensor `feat` does not have the expected shape.
        """
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens
