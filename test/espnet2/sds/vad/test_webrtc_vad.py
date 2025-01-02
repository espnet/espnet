import pytest
import soundfile

from espnet2.sds.vad.webrtc_vad import WebrtcVADModel

pytest.importorskip("webrtcvad")


def test_forward():
    vad_model = WebrtcVADModel()
    vad_model.warmup()
    x, rate = soundfile.read("test_utils/ctc_align_test.wav", dtype="int16")
    vad_model.forward(x, rate)
    vad_model.forward(x, rate, binary=True)
