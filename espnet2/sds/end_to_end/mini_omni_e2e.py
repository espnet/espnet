import base64
import io
import tempfile
from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.sds.end_to_end.abs_e2e import AbsE2E

try:
    from pydub import AudioSegment

    is_pydub_available = True
except ImportError:
    is_pydub_available = False


class MiniOmniE2EModel(AbsE2E):
    """Mini-OMNI E2E"""

    @typechecked
    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """A class to initialize and manage the OmniInference client

        for end-to-end dialogue systems.

        Args:
            device (Literal["cuda", "cpu"], optional):
                The device to run the inference on. Defaults to "cuda".

        Raises:
            ImportError:
                If required dependencies (Pydub, Huggingface Hub,
                or OmniInference) are not installed.
        """
        if not is_pydub_available:
            raise ImportError("Error: Pydub is not properly installed.")
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            print("Error: Huggingface_hub is not properly installed.")
            raise e
        try:
            from espnet2.sds.end_to_end.mini_omni.inference import OmniInference
        except Exception as e:
            print(
                "Error: Dependencies not properly installed."
                "Check https://huggingface.co/spaces/gradio/"
                "omni-mini/blob/main/requirements.txt"
            )
            raise e

        super().__init__()
        repo_id = "gpt-omni/mini-omni"
        snapshot_download(repo_id, local_dir="./checkpoint", revision="main")

        self.client = OmniInference("./checkpoint", "cuda")
        self.stream_stride = 4
        self.max_tokens = 2048
        self.OUT_CHANNELS = 1
        self.OUT_RATE = 24000
        self.OUT_SAMPLE_WIDTH = 2
        self.device = device
        self.dtype = dtype

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
        dummy_input = (
            torch.randn(
                (3000),
                dtype=getattr(torch, self.dtype),
                device="cpu",
            )
            .cpu()
            .numpy()
        )
        array = dummy_input
        orig_sr = 16000
        audio_buffer = io.BytesIO()
        segment = AudioSegment(
            array.tobytes(),
            frame_rate=orig_sr,
            sample_width=array.dtype.itemsize,
            channels=(1 if len(array.shape) == 1 else array.shape[1]),
        )
        segment.export(audio_buffer, format="wav")
        base64_encoded = str(
            base64.b64encode(audio_buffer.getvalue()), encoding="utf-8"
        )
        data_buff = base64.b64decode(base64_encoded.encode("utf-8"))
        stream_stride = 4
        max_tokens = 2048

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data_buff)
            audio_generator = self.client.run_AT_batch_stream(
                f.name, stream_stride, max_tokens
            )
        _ = [k for k in audio_generator]

    def forward(
        self,
        array: np.ndarray,
        orig_sr: int,
    ) -> Tuple[str, bytes]:
        """Processes audio input to generate synthesized speech

        and the corresponding text response.

        Args:
            array (np.ndarray):
                The input audio array to be processed.
            orig_sr (int):
                The sample rate of the input audio.

        Returns:
            Tuple[str, bytes]:
                A tuple containing:
                - `text_str` (str): The generated text response.
                - `audio_output` (bytes): The synthesized speech
                as an MP3 byte stream.
        """
        audio_buffer = io.BytesIO()
        segment = AudioSegment(
            array.tobytes(),
            frame_rate=orig_sr,
            sample_width=array.dtype.itemsize,
            channels=(1 if len(array.shape) == 1 else array.shape[1]),
        )
        segment.export(audio_buffer, format="wav")
        base64_encoded = str(
            base64.b64encode(audio_buffer.getvalue()), encoding="utf-8"
        )
        data_buff = base64.b64decode(base64_encoded.encode("utf-8"))
        stream_stride = 4
        max_tokens = 2048

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data_buff)
            audio_generator = self.client.run_AT_batch_stream(
                f.name, stream_stride, max_tokens
            )
        ans = [k for k in audio_generator]
        text_str = ans[-1]
        ans = ans[:-1]
        output_buffer = b""
        for chunk in ans:
            audio_segment = AudioSegment(
                chunk,
                frame_rate=self.OUT_RATE,
                sample_width=self.OUT_SAMPLE_WIDTH,
                channels=self.OUT_CHANNELS,
            )
            mp3_io = io.BytesIO()
            audio_segment.export(mp3_io, format="mp3", bitrate="320k")
            mp3_bytes = mp3_io.getvalue()
            mp3_io.close()
            output_buffer += mp3_bytes
        audio_output = output_buffer
        return (text_str, audio_output)
