from typing import Optional

import librosa
import numpy as np
import torch
from typeguard import typechecked

from espnet2.sds.utils.utils import int2float
from espnet2.sds.vad.abs_vad import AbsVAD

try:
    import webrtcvad

    is_webrtcvad_available = True
except ImportError:
    is_webrtcvad_available = False


class WebrtcVADModel(AbsVAD):
    """Webrtc VAD Model"""

    @typechecked
    def __init__(
        self,
        speakup_threshold: int = 12,
        continue_threshold: int = 10,
        min_speech_ms: int = 500,
        max_speech_ms: float = float("inf"),
        target_sr: int = 16000,
    ):
        """This class uses WebRTC VAD to detect speech in an audio stream.

        Args:
            speakup_threshold (int, optional):
                The threshold for detecting the start of speech.
            continue_threshold (int, optional):
                The threshold for continuing speech detection.
            min_speech_ms (int, optional):
                The minimum duration (in milliseconds) for a valid
                speech segment. Defaults to 500 ms.
            max_speech_ms (float, optional):
                The maximum duration (in milliseconds) for a valid
                speech segment. Defaults to infinity.
            target_sr (int, optional):
                The target sampling rate for resampling the input
                audio. Defaults to 16000 Hz.

        Attributes:
            vad_output (Optional[list]):
                Stores the speech segments detected as
                floating-point tensors.
            vad_bin_output (Optional[list]):
                Stores the speech segments detected as binary audio.

        Raises:
            ImportError:
                If the required `webrtcvad` library is not installed.
        """
        if not is_webrtcvad_available:
            raise ImportError("Error: webrtcvad is not properly installed.")
        super().__init__()
        self.vad_output = None
        self.vad_bin_output = None
        self.speakup_threshold = speakup_threshold
        self.continue_threshold = continue_threshold
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.target_sr = target_sr

    def warmup(self):
        return

    def forward(
        self,
        speech: np.ndarray,
        sample_rate: int,
        binary: bool = False,
    ) -> Optional[np.ndarray]:
        """Process an audio stream and detect speech using WebRTC VAD.

        Args:
            speech:
                The raw audio stream in 16-bit PCM format.
            sample_rate (int):
                The sampling rate of the input audio.
            binary (bool, optional):
                If True, returns the binary audio output instead of the
                resampled float array. Defaults to False.

        Returns:
            Optional[np.ndarray]:
                The detected speech segment as a NumPy array
                (float or binary audio), or None if no valid segment is found.
        """
        audio_int16 = np.frombuffer(speech, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        audio_float32 = librosa.resample(
            audio_float32, orig_sr=sample_rate, target_sr=self.target_sr
        )
        vad_count = 0
        chunk_size = int(320 * sample_rate / 16000)
        for i in range(int(len(speech) / chunk_size)):
            vad = webrtcvad.Vad()
            vad.set_mode(3)
            if vad.is_speech(
                speech[i * chunk_size : (i + 1) * chunk_size].tobytes(), sample_rate
            ):
                vad_count += 1
        if self.vad_output is None and vad_count > self.speakup_threshold:
            vad_curr = True
            self.vad_output = [torch.from_numpy(audio_float32)]
            self.vad_bin_output = [speech]
        elif self.vad_output is not None and vad_count > self.continue_threshold:
            vad_curr = True
            self.vad_output.append(torch.from_numpy(audio_float32))
            self.vad_bin_output.append(speech)
        else:
            vad_curr = False
        if self.vad_output is not None and vad_curr is False:
            array = torch.cat(self.vad_output).cpu().numpy()
            duration_ms = len(array) / self.target_sr * 1000
            if not (
                duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms
            ):
                if binary:
                    array = np.concatenate(self.vad_bin_output)
                self.vad_output = None
                self.vad_bin_output = None
                return array
        return None
