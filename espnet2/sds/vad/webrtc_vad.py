from espnet2.sds.vad.abs_vad import AbsVAD
from espnet2.sds.utils.utils import int2float
import os
import numpy as np
from typeguard import typechecked
import torch
import librosa

try:
    import webrtcvad
except Exception as e:
    print("Error: WebRTC is not properly installed.")
    raise e

class WebrtcVADModel(AbsVAD):
    """Webrtc VAD Model"""

    @typechecked
    def __init__(
        self,
        speakup_threshold=12,
        continue_threshold=10,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        target_sr=16000,
    ): 
        super().__init__()
        self.vad_output=None
        self.speakup_threshold=speakup_threshold
        self.continue_threshold=continue_threshold
        self.min_speech_ms=min_speech_ms
        self.max_speech_ms=max_speech_ms
        self.target_sr=target_sr

    
    def warmup(self):
        return
    
    def forward(self,speech,sample_rate):
        audio_int16 = np.frombuffer(speech, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        audio_float32=librosa.resample(audio_float32, orig_sr=sample_rate, target_sr=self.target_sr)
        vad_count=0
        for i in range(int(len(speech)/960)):
            vad = webrtcvad.Vad()
            vad.set_mode(3)
            if (vad.is_speech(speech[i*960:(i+1)*960].tobytes(), sample_rate)):
                vad_count+=1
        if self.vad_output is None and vad_count>self.speakup_threshold:
            vad_curr=True
            self.vad_output=[torch.from_numpy(audio_float32)]
        elif self.vad_output is not None and vad_count>self.continue_threshold:
            vad_curr=True
            self.vad_output.append(torch.from_numpy(audio_float32))
        else:
            vad_curr=False
        if self.vad_output is not None and vad_curr==False:
            array = torch.cat(self.vad_output).cpu().numpy()
            duration_ms = len(array) / self.target_sr * 1000
            if (not(duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms)):
                self.vad_output=None
                return array
        return None
