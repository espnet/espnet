import pytest
import soundfile
import torch

from espnet2.sds.espnet_model import ESPnetSDSModelInterface

pytest.importorskip("gradio")


def test_forward():
    pytest.importorskip("webrtcvad")
    if not torch.cuda.is_available():
        return  # Only GPU supported
    dialogue_model = ESPnetSDSModelInterface(
        ASR_option="librispeech_asr",
        LLM_option="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TTS_option="kan-bayashi/ljspeech_vits",
        type_option="Cascaded",
        access_token="",
    )
    x, rate = soundfile.read("test_utils/ctc_align_test.wav", dtype="int16")
    gen = dialogue_model.handle_type_selection(
        option="Cascaded",
        TTS_radio="kan-bayashi/ljspeech_vits",
        ASR_radio="librispeech_asr",
        LLM_radio="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )
    for _ in gen:
        continue
    dialogue_model.forward(
        x,
        rate,
        x,
        asr_output_str=None,
        text_str=None,
        audio_output=None,
        audio_output1=None,
        latency_ASR=0.0,
        latency_LM=0.0,
        latency_TTS=0.0,
    )


def test_handle_E2E_selection():
    pytest.importorskip("pydub")
    pytest.importorskip("espnet2.sds.end_to_end.mini_omni.inference")
    pytest.importorskip("huggingface_hub")
    if not torch.cuda.is_available():
        return  # Only GPU supported
    dialogue_model = ESPnetSDSModelInterface(
        ASR_option="librispeech_asr",
        LLM_option="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TTS_option="kan-bayashi/ljspeech_vits",
        type_option="Cascaded",
        access_token="",
    )
    x, rate = soundfile.read("test_utils/ctc_align_test.wav", dtype="int16")
    dialogue_model.handle_type_selection(
        option="E2E",
        TTS_radio="kan-bayashi/ljspeech_vits",
        ASR_radio="librispeech_asr",
        LLM_radio="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )
    assert dialogue_model.text2speech is None
    assert dialogue_model.s2t is None
    assert dialogue_model.LM_pipe is None
    assert dialogue_model.ASR_curr_name is None
    assert dialogue_model.LLM_curr_name is None
    assert dialogue_model.TTS_curr_name is None
    dialogue_model.forward(
        x,
        rate,
        x,
        asr_output_str=None,
        text_str=None,
        audio_output=None,
        audio_output1=None,
        latency_ASR=0.0,
        latency_LM=0.0,
        latency_TTS=0.0,
    )
    dialogue_model.handle_type_selection(
        option="Cascaded",
        TTS_radio="kan-bayashi/ljspeech_vits",
        ASR_radio="librispeech_asr",
        LLM_radio="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )
    assert dialogue_model.client is None
