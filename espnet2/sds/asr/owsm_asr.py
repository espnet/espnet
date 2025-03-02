import librosa
import numpy as np
import torch
from typeguard import typechecked

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.sds.asr.abs_asr import AbsASR


class OWSMModel(AbsASR):
    """OWSM ASR"""

    @typechecked
    def __init__(
        self,
        tag: str = "espnet/owsm_v3.1_ebf",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """Initializer method.

        Args:
        tag (str, optional):
            The pre-trained model tag (on Hugging Face).
            Defaults to:
            "espnet/owsm_v3.1_ebf".
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
            task_sym="<asr>",
            beam_size=1,
            predict_time=False,
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
            speech = librosa.util.fix_length(dummy_input, size=(16000 * 30))
            _ = self.s2t(speech)

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
                as returned by the OWSM ASR model.
        """
        with torch.no_grad():
            array = librosa.util.fix_length(array, size=(16000 * 30))
            prompt = " ".join(self.s2t(array)[0][0].split()[1:])
            return prompt
