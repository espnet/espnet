import librosa  # noqa
import numpy as np
import torch
from typeguard import typechecked

from espnet2.sds.asr.abs_asr import AbsASR


class WhisperASRModel(AbsASR):
    """Whisper ASR"""

    @typechecked
    def __init__(
        self,
        tag: str = "large",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """Initializer method.

        Args:
        tag (str, optional):
            The Whisper model tag
        device (str, optional):
            The computation device for running inference.
            Defaults to "cuda".
            Common options include "cuda" or "cpu".
        dtype (str, optional):
            The floating-point precision to use.
            Defaults to "float16".
        """
        super().__init__()
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e
        self.s2t = whisper.load_model(tag, device=device)
        self.device = device
        self.dtype = dtype

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
        with torch.no_grad():
            dummy_input = (
                torch.randn(
                    (3000),
                    dtype=getattr(torch, self.dtype),
                    device="cpu",
                )
                .cpu()
                .numpy()
            )
            _ = self.s2t.transcribe(torch.tensor(dummy_input).float(), beam_size=1)[
                "text"
            ]

    def forward(self, array: np.ndarray) -> str:
        """Perform a forward pass on the given audio data,

        returning the transcribed text prompt.

        Args:
            array (np.ndarray):
                The input audio data to be transcribed.
                Typically a NumPy array.

        Returns:
            str:
                The transcribed text from the audio input,
                as returned by the Whisper ASR model.
        """
        with torch.no_grad():
            prompt = self.s2t.transcribe(torch.tensor(array).float(), beam_size=1)[
                "text"
            ]
            return prompt
