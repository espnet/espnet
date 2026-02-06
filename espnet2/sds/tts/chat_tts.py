from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.sds.tts.abs_tts import AbsTTS


class ChatTTSModel(AbsTTS):
    """ChaTTS Model"""

    @typechecked
    def __init__(
        self,
    ):
        """Initializes the ChatTTSModel class.

        Ensures that the `ChatTTS` library is properly installed
        and initializes the TTS engine.
        """
        super().__init__()
        try:
            import ChatTTS
        except Exception as e:
            print("Error: ChatTTS is not properly installed.")
            raise e
        self.text2speech = ChatTTS.Chat()
        self.text2speech.load(compile=False)

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
        with torch.no_grad():
            _ = self.text2speech.infer(["Sid"])[0]

    def forward(self, transcript: str) -> Tuple[int, np.ndarray]:
        """Converts a text transcript into an audio waveform

        using the ChatTTS system.

        Args:
            transcript (str):
                The input text to be converted into speech.

        Returns:
            Tuple[int, np.ndarray]:
                A tuple containing:
                - The sample rate of the audio (int).
                - The generated audio waveform as a
                NumPy array of type `int16`.
        """
        with torch.no_grad():
            audio_chunk = self.text2speech.infer([transcript])[0]
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            return (24000, audio_chunk)
