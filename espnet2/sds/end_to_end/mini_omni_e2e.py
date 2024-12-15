from espnet2.sds.end_to_end.abs_e2e import AbsE2E
import os
import numpy as np
import torch
from typeguard import typechecked
from huggingface_hub import snapshot_download
import io
from pydub import AudioSegment
import base64
import tempfile

class MiniOmniE2EModel(AbsE2E):
    """Mini-OMNI E2E"""

    @typechecked
    def __init__(
        self,
        device="cuda",
    ): 
        super().__init__()
        repo_id = "gpt-omni/mini-omni"
        snapshot_download(repo_id, local_dir="./checkpoint", revision="main")
        from espnet2.sds.end_to_end.mini_omni.inference import OmniInference
        self.client = OmniInference("./checkpoint", "cuda")
        self.stream_stride = 4
        self.max_tokens = 2048
        self.OUT_CHANNELS = 1
        self.OUT_RATE = 24000
        self.OUT_SAMPLE_WIDTH = 2
        self.device=device
    
    def warmup(self):
        dummy_input = torch.randn(
                (3000),
                dtype=getattr(torch, "float16"),
                device="cpu",
        ).cpu().numpy()
        array=dummy_input
        orig_sr=16000
        audio_buffer = io.BytesIO()
        segment = AudioSegment(array.tobytes(),frame_rate=orig_sr,sample_width=array.dtype.itemsize, channels=(1 if len(array.shape) == 1 else array.shape[1]),)
        segment.export(audio_buffer, format="wav")
        base64_encoded = str(base64.b64encode(audio_buffer.getvalue()), encoding="utf-8")
        data_buff = base64.b64decode(base64_encoded.encode("utf-8"))
        stream_stride = 4
        max_tokens = 2048

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data_buff)
            audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
        ans=[k for k in audio_generator]
    
    def forward(self,array, orig_sr):
        audio_buffer = io.BytesIO()
        segment = AudioSegment(array.tobytes(),frame_rate=orig_sr,sample_width=array.dtype.itemsize, channels=(1 if len(array.shape) == 1 else array.shape[1]),)
        segment.export(audio_buffer, format="wav")
        base64_encoded = str(base64.b64encode(audio_buffer.getvalue()), encoding="utf-8")
        data_buff = base64.b64decode(base64_encoded.encode("utf-8"))
        stream_stride = 4
        max_tokens = 2048

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data_buff)
            audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
        ans=[k for k in audio_generator]
        text_str=ans[-1]
        ans=ans[:-1]
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
        audio_output=output_buffer
        return (text_str, audio_output)