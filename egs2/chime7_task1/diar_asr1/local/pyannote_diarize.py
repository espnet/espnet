from pyannote.audio import Pipeline
import soundfile as sf
import torch


def diarize_session(pipeline, wav_file, uem_boundaries=None):

    audio, fs = sf.read(wav_file)
    uem_boundaries = [round(x*fs) for x in uem_boundaries]
    assert audio.ndim == 1, "Multi-channel audio not supported right now."
    audio = audio[uem_boundaries[0]:uem_boundaries[1]]

    result = pipeline({"waveform": })
