import librosa  # noqa
import numpy as np
import torch
from typeguard import typechecked

from espnet2.bin.asr_inference import Speech2Text
from espnet2.sds.asr.abs_asr import AbsASR


class ESPnetASRModel(AbsASR):
    """ESPnet ASR"""

    @typechecked
    def __init__(
        self,
        tag: str = (
            "espnet/"
            "simpleoier_librispeech_asr_train_asr_conformer7_"
            "wavlm_large_raw_en_bpe5000_sp"
        ),
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """Initializer method.

        Args:
        tag (str, optional):
            The pre-trained model tag (on Hugging Face).
            Defaults to:
            "espnet/simpleoier_librispeech_asr_train_asr_
            conformer7_wavlm_large_raw_en_bpe5000_sp".
        device (str, optional):
            The computation device for running inference.
            Defaults to "cuda".
            Common options include "cuda" or "cpu".
        dtype (str, optional):
            The floating-point precision to use.
            Defaults to "float16".
        """
        super().__init__()
        self.s2t = Speech2Text.from_pretrained(
            model_tag=tag,
            device=device,
            beam_size=1,
        )
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
            _ = self.s2t(dummy_input)[0][0]

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
                as returned by the speech-to-text model.
        """
        with torch.no_grad():
            prompt = self.s2t(array)[0][0]
            return prompt
