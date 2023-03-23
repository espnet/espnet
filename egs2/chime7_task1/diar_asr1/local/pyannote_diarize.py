from pyannote.audio import Pipeline
import soundfile as sf
import torch
from pathlib import Path


def rttm2json(rttm_file):
    with open(rttm_file, "r") as f:
        rttm = f.readlines()

    rttm = [x.rstrip("\n") for x in rttm]
    filename = Path(rttm_file).stem

    to_json = []
    for line in rttm:
        current = line.split(" ")
        start = current[3]
        duration = current[4]
        stop = str(float(start) + float(duration))
        speaker = current[7]
        session = filename
        to_json.append(
            {
                "session_id": session,
                "speaker": speaker,
                "start_time": start,
                "end_time": stop,
            }
        )

    to_json = sorted(to_json, key=lambda x: x["start_time"])
    return to_json


def apply_doverlap():
    pass

def diarize_session(pipeline, wav_file, uem_boundaries=None):

    audio, fs = sf.read(wav_file, dtype="float32")
    uem_boundaries = [round(x*fs) for x in uem_boundaries]
    assert audio.ndim == 1, "Multi-channel audio not supported right now."
    # cut out portions outside uem
    audio = audio[uem_boundaries[0]:uem_boundaries[1]]

    import pdb
    pdb.set_trace()
    result = pipeline({"waveform": torch.from_numpy(audio[None, ...]), "sample_rate": fs})
    # result is an annotation


if __name__ == "__main__":
    pipeline = Pipeline()
    wav_file = "/home/samco/dgx/CHiME6/decode_chime7/espnet/egs2/chime7_task1/asr1/chime7_task1/mixer6/audio/dev/20090714_134807_LDC_120290_CH01.flac"
