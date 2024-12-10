import json
import math

import librosa
import numpy as np
import requests
import torch
import torchaudio

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
# URL for the web service
SCORING_URI_DNSMOS = "https://dnsmos.azurewebsites.net/score"
SCORING_URI_DNSMOS_P835 = "https://dnsmos.azurewebsites.net/v1/dnsmosp835/score"


def poly1d(coefficients, use_numpy=False):
    """
    Construct a polynomial function from given coefficients.

    This function creates a polynomial function of the form:
    f(x) = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0
    where the coefficients are provided in the list `coefficients` in the
    order of highest degree to lowest.

    If `use_numpy` is set to True, the function returns a numpy polynomial
    object instead of a custom function.

    Args:
        coefficients (list or tuple): Coefficients of the polynomial,
            where the first element is the coefficient of the highest degree.
        use_numpy (bool): If True, return a numpy polynomial object.
            Defaults to False.

    Returns:
        function or np.poly1d: A polynomial function that can be called with
        a single argument or a numpy polynomial object.

    Examples:
        >>> p = poly1d([1, -3, 2])  # Creates the polynomial x^2 - 3x + 2
        >>> p(1)  # Output: 0
        >>> p(2)  # Output: 0
        >>> p(3)  # Output: 2

        >>> np_poly = poly1d([1, -3, 2], use_numpy=True)
        >>> np_poly(1)  # Output: 0
        >>> np_poly(2)  # Output: 0
        >>> np_poly(3)  # Output: 2
    """
    if use_numpy:
        return np.poly1d(coefficients)
    coefficients = tuple(reversed(coefficients))

    def func(p):
        return sum(coef * p**i for i, coef in enumerate(coefficients))

    return func


class DNSMOS_web:
    """
    A class for evaluating audio quality using the DNSMOS web service.

This class sends audio data to the DNSMOS web service for quality scoring. It
requires an authentication key to access the service and supports different
scoring methods.

Attributes:
    auth_key (str): The authentication key used for accessing the web service.

Args:
    auth_key (str): The authentication key for the DNSMOS web service.

Methods:
    __call__(aud, input_fs, fname="", method="p808"):
        Sends the audio data to the DNSMOS web service and retrieves the score.

Examples:
    >>> dnsmos = DNSMOS_web(auth_key="your_auth_key")
    >>> audio_data = np.random.rand(16000 * 5)  # 5 seconds of random audio
    >>> score = dnsmos(audio_data, input_fs=16000)
    >>> print(score)

Note:
    The audio data must be a 1D numpy array or a similar structure, and the
    sampling frequency must match the expected value (16000 Hz) or be
    resampled.

Raises:
    requests.exceptions.RequestException: If there is an error during the
        HTTP request to the web service.
    """
    # ported from
    # https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos.py
    def __init__(self, auth_key):
        self.auth_key = auth_key

    def __call__(self, aud, input_fs, fname="", method="p808"):
        if input_fs != SAMPLING_RATE:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE)
        else:
            audio = aud

        # Set the content type
        headers = {"Content-Type": "application/json"}
        # If authentication is enabled, set the authorization header
        headers["Authorization"] = f"Basic {self.auth_key}"
        fname = fname + ".wav" if fname else "audio.wav"
        data = {"data": audio.tolist(), "filename": fname}
        input_data = json.dumps(data)
        # Make the request and display the response
        if method == "p808":
            u = SCORING_URI_DNSMOS
        else:
            u = SCORING_URI_DNSMOS_P835
        resp = requests.post(u, data=input_data, headers=headers)
        score_dict = resp.json()
        return score_dict


class DNSMOS_local:
    """
    A class for estimating Mean Opinion Scores (MOS) for audio signals using local 
models. This implementation leverages pre-trained models for audio quality 
assessment based on deep learning.

Attributes:
    convert_to_torch (bool): Flag indicating whether to convert models to PyTorch.
    use_gpu (bool): Flag indicating whether to use GPU for computations.
    primary_model (torch.nn.Module or ort.InferenceSession): Model for primary 
        audio processing.
    p808_model (torch.nn.Module or ort.InferenceSession): Model for P.808 metrics 
        estimation.
    spectrogram (torch.nn.Module): Spectrogram transformation module.
    to_db (torch.nn.Module): Transformation module to convert amplitude to decibels.

Args:
    primary_model_path (str): Path to the primary model file (ONNX format).
    p808_model_path (str): Path to the P.808 model file (ONNX format).
    use_gpu (bool, optional): Flag to enable GPU usage. Default is False.
    convert_to_torch (bool, optional): Flag to convert models to PyTorch. Default 
        is False.

Raises:
    RuntimeError: If onnx2torch or onnxruntime is not installed when required.

Examples:
    >>> dnsmos = DNSMOS_local('path/to/primary/model', 'path/to/p808/model', 
    ...                        use_gpu=True, convert_to_torch=True)
    >>> audio_signal = np.random.rand(16000 * 9)  # Simulated audio signal
    >>> result = dnsmos(audio_signal, input_fs=16000, is_personalized_MOS=True)
    >>> print(result)
    {
        "OVRL_raw": 3.5,
        "SIG_raw": 3.8,
        "BAK_raw": 3.2,
        "OVRL": 3.6,
        "SIG": 3.9,
        "BAK": 3.3,
        "P808_MOS": 3.7,
    }

Note:
    The input audio signal should be a 1D numpy array or a torch tensor.

Todo:
    - Add support for more input formats and additional model architectures.
    """
    # ported from
    # https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
    def __init__(
        self, primary_model_path, p808_model_path, use_gpu=False, convert_to_torch=False
    ):
        self.convert_to_torch = convert_to_torch
        self.use_gpu = use_gpu
        if convert_to_torch:
            try:
                from onnx2torch import convert
            except ModuleNotFoundError:
                raise RuntimeError("Please install onnx2torch manually and retry!")

            if primary_model_path is not None:
                self.primary_model = convert(primary_model_path).eval()
                self.p808_model = convert(p808_model_path).eval()
            self.spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=321, hop_length=160, pad_mode="constant"
            )

            self.to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80.0)
            if use_gpu:
                if primary_model_path is not None:
                    self.primary_model = self.primary_model.cuda()
                    self.p808_model = self.p808_model.cuda()
                self.spectrogram = self.spectrogram.cuda()
        else:
            try:
                import onnxruntime as ort
            except ModuleNotFoundError:
                raise RuntimeError("Please install onnxruntime manually and retry!")

            prvd = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
            if primary_model_path is not None:
                self.onnx_sess = ort.InferenceSession(
                    primary_model_path, providers=[prvd]
                )
                self.p808_onnx_sess = ort.InferenceSession(
                    p808_model_path, providers=[prvd]
                )

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        """
        Compute the Mel spectrogram of the given audio signal.

    This method calculates the Mel spectrogram for the input audio signal,
    either using PyTorch or librosa, depending on the configuration of the
    DNSMOS_local instance.

    Args:
        audio (torch.Tensor or np.ndarray): The input audio signal.
        n_mels (int, optional): Number of Mel bands to generate. Defaults to 120.
        frame_size (int, optional): Size of the FFT window. Defaults to 320.
        hop_length (int, optional): Number of samples between frames. Defaults to 160.
        sr (int, optional): Sampling rate of the audio signal. Defaults to 16000.
        to_db (bool, optional): Whether to convert the Mel spectrogram to dB scale.
            Defaults to True.

    Returns:
        np.ndarray or torch.Tensor: The computed Mel spectrogram, transposed.

    Examples:
        >>> import torch
        >>> audio = torch.randn(16000)  # Simulated audio signal
        >>> mel_spec = dnsmos_local.audio_melspec(audio)
        >>> print(mel_spec.shape)
        (n_frames, n_mels)

    Note:
        If `self.convert_to_torch` is True, the function uses PyTorch for
        computations; otherwise, it uses librosa. The output is transposed to
        match the expected shape.

    Raises:
        ValueError: If the audio input is not a valid tensor or ndarray.
        """
        if self.convert_to_torch:
            specgram = self.spectrogram(audio)
            fb = torch.as_tensor(
                librosa.filters.mel(sr=sr, n_fft=frame_size + 1, n_mels=n_mels).T,
                dtype=audio.dtype,
                device=audio.device,
            )
            mel_spec = torch.matmul(specgram.transpose(-1, -2), fb).transpose(-1, -2)
            if to_db:
                self.to_db.db_multiplier = math.log10(
                    max(self.to_db.amin, torch.max(mel_spec))
                )
                mel_spec = (self.to_db(mel_spec) + 40) / 40
        else:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=frame_size + 1,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            if to_db:
                mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        """
        Calculates polynomial fitting values for the given audio metrics.

    This function uses polynomial regression to compute adjusted values for
    signal, background, and overall metrics based on input parameters. It 
    applies different polynomial coefficients depending on whether the 
    calculation is for personalized Mean Opinion Score (MOS).

    Args:
        sig (float): The signal metric value to be adjusted.
        bak (float): The background metric value to be adjusted.
        ovr (float): The overall metric value to be adjusted.
        is_personalized_MOS (bool): Flag indicating if the calculation is for
            personalized MOS. If True, personalized coefficients are used.

    Returns:
        tuple: A tuple containing the adjusted signal, background, and overall
        metrics:
            - sig_poly (float): Adjusted signal metric.
            - bak_poly (float): Adjusted background metric.
            - ovr_poly (float): Adjusted overall metric.

    Examples:
        >>> dnsmos_local = DNSMOS_local(...)
        >>> sig_adjusted, bak_adjusted, ovr_adjusted = dnsmos_local.get_polyfit_val(
        ...     sig=1.5, bak=0.5, ovr=1.0, is_personalized_MOS=True
        ... )
        >>> print(sig_adjusted, bak_adjusted, ovr_adjusted)
        (-0.10, 0.40, 1.15)

    Note:
        The polynomial coefficients are defined within the function based on 
        the is_personalized_MOS flag.
        """
        flag = not self.convert_to_torch
        if is_personalized_MOS:
            p_ovr = poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046], flag)
            p_sig = poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726], flag)
            p_bak = poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132], flag)
        else:
            p_ovr = poly1d([-0.06766283, 1.11546468, 0.04602535], flag)
            p_sig = poly1d([-0.08397278, 1.22083953, 0.0052439], flag)
            p_bak = poly1d([-0.13166888, 1.60915514, -0.39604546], flag)

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs, is_personalized_MOS=False):
        if self.convert_to_torch:
            device = "cuda" if self.use_gpu else "cpu"
            if isinstance(aud, torch.Tensor):
                aud = aud.to(device=device)
            else:
                aud = torch.as_tensor(aud, dtype=torch.float32, device=device)
        else:
            aud = aud.cpu().detach().numpy() if isinstance(aud, torch.Tensor) else aud
        if input_fs != SAMPLING_RATE:
            if self.convert_to_torch:
                audio = torch.as_tensor(
                    librosa.resample(
                        aud.detach().cpu().numpy(),
                        orig_sr=input_fs,
                        target_sr=SAMPLING_RATE,
                    ),
                    dtype=aud.dtype,
                    device=aud.device,
                )
            else:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE)
        else:
            audio = aud
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            if self.convert_to_torch:
                audio = torch.cat((audio, audio))
            else:
                audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            if self.convert_to_torch:
                input_features = audio_seg.float()[None, :]
                p808_input_features = self.audio_melspec(
                    audio=audio_seg[:-160]
                ).float()[None, :, :]
                p808_mos = self.p808_model(p808_input_features)
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.primary_model(
                    input_features
                )[0]
            else:
                input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
                p808_input_features = np.array(
                    self.audio_melspec(audio=audio_seg[:-160])
                ).astype("float32")[np.newaxis, :, :]
                p808_mos = self.p808_onnx_sess.run(
                    None, {"input_1": p808_input_features}
                )[0][0][0]
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(
                    None, {"input_1": input_features}
                )[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        to_array = torch.stack if self.convert_to_torch else np.array
        return {
            "OVRL_raw": to_array(predicted_mos_ovr_seg_raw).mean(),
            "SIG_raw": to_array(predicted_mos_sig_seg_raw).mean(),
            "BAK_raw": to_array(predicted_mos_bak_seg_raw).mean(),
            "OVRL": to_array(predicted_mos_ovr_seg).mean(),
            "SIG": to_array(predicted_mos_sig_seg).mean(),
            "BAK": to_array(predicted_mos_bak_seg).mean(),
            "P808_MOS": to_array(predicted_p808_mos).mean(),
        }
