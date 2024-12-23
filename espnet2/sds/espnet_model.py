import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.sds.asr.espnet_asr import ESPnetASRModel
from espnet2.sds.asr.owsm_asr import OWSMModel
from espnet2.sds.asr.owsm_ctc_asr import OWSMCTCModel
from espnet2.sds.asr.whisper_asr import WhisperASRModel
from espnet2.sds.end_to_end.mini_omni_e2e import MiniOmniE2EModel
from espnet2.sds.llm.hugging_face_llm import HuggingFaceLLM
from espnet2.sds.tts.chat_tts import ChatTTSModel
from espnet2.sds.tts.espnet_tts import ESPnetTTSModel
from espnet2.sds.utils.chat import Chat
from espnet2.sds.vad.webrtc_vad import WebrtcVADModel
from espnet2.train.abs_espnet_model import AbsESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSDSModelInterface(AbsESPnetModel):
    """Web Interface for Spoken Dialog System models"""

    @typechecked
    def __init__(self, ASR_option, LLM_option, TTS_option, type_option, access_token):
        super().__init__()
        self.TTS_option = TTS_option
        self.ASR_option = ASR_option
        self.LLM_option = LLM_option
        self.type_option = type_option
        self.access_token = access_token
        self.TTS_curr_name = None
        self.LLM_curr_name = None
        self.ASR_curr_name = None
        self.text2speech = None
        self.s2t = None
        self.LM_pipe = None
        self.client = None
        self.vad_model = WebrtcVADModel()
        self.chat = Chat(2)
        self.chat.init_chat(
            {
                "role": "system",
                "content": "You are a helpful and friendly AI assistant. The user is talking to you with their voice and you should respond in a conversational style. You are polite, respectful, and aim to provide concise and complete responses of less than 15 words.",
            }
        )
        self.user_role = "user"

    def handle_TTS_selection(self, option):
        if self.TTS_curr_name is not None:
            if option == self.TTS_curr_name:
                return
        yield gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Audio(
            visible=False
        )
        self.TTS_curr_name = option
        tag = option
        if tag == "ChatTTS":
            self.text2speech = ChatTTSModel()
        else:
            self.text2speech = ESPnetTTSModel(tag)
        self.text2speech.warmup()
        yield gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Audio(visible=True)

    def handle_LLM_selection(self, option):
        if self.LLM_curr_name is not None:
            if option == self.LLM_curr_name:
                return
        yield gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Audio(
            visible=False
        )
        self.LLM_curr_name = option
        self.LM_pipe = HuggingFaceLLM(access_token=self.access_token, tag=option)
        self.LM_pipe.warmup()
        yield gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Audio(visible=True)

    def handle_ASR_selection(self, option):
        if option == "librispeech_asr":
            option = "espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp"
        if self.ASR_curr_name is not None:
            if option == self.ASR_curr_name:
                return
        yield gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Audio(
            visible=False
        )
        self.ASR_curr_name = option
        if option == "espnet/owsm_v3.1_ebf":
            self.s2t = OWSMModel()
        elif (
            option
            == "espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp"
        ):
            self.s2t = ESPnetASRModel(tag=option)
        elif option == "whisper":
            self.s2t = WhisperASRModel()
        else:
            self.s2t = OWSMCTCModel(tag=option)

        self.s2t.warmup()
        yield gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Audio(visible=True)

    def handle_E2E_selection(self):
        if self.client is None:
            self.client = MiniOmniE2EModel()
            self.client.warmup()

    def handle_type_selection(self, option, TTS_radio, ASR_radio, LLM_radio):
        yield (
            gr.Radio(visible=False),
            gr.Radio(visible=False),
            gr.Radio(visible=False),
            gr.Radio(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Audio(visible=False),
            gr.Radio(visible=False),
            gr.Radio(visible=False),
        )
        if option == "Cascaded":
            self.client = None
            for _ in self.handle_TTS_selection(TTS_radio):
                continue
            for _ in self.handle_ASR_selection(ASR_radio):
                continue
            for _ in self.handle_LLM_selection(LLM_radio):
                continue
            yield (
                gr.Radio(visible=True),
                gr.Radio(visible=True),
                gr.Radio(visible=True),
                gr.Radio(visible=False),
                gr.Textbox(visible=True),
                gr.Textbox(visible=True),
                gr.Audio(visible=True),
                gr.Radio(visible=True, interactive=True),
                gr.Radio(visible=False),
            )
        else:
            self.text2speech = None
            self.s2t = None
            self.LM_pipe = None
            self.ASR_curr_name = None
            self.LLM_curr_name = None
            self.TTS_curr_name = None
            self.handle_E2E_selection()
            yield (
                gr.Radio(visible=False),
                gr.Radio(visible=False),
                gr.Radio(visible=False),
                gr.Radio(visible=True),
                gr.Textbox(visible=True),
                gr.Textbox(visible=True),
                gr.Audio(visible=True),
                gr.Radio(visible=False),
                gr.Radio(visible=True, interactive=True),
            )

    def forward(
        self,
        y,
        sr,
        stream,
        asr_output_str,
        text_str,
        audio_output,
        audio_output1,
        latency_ASR,
        latency_LM,
        latency_TTS,
    ):
        orig_sr = sr
        sr = 16000
        if self.client is not None:
            array = self.vad_model(y, orig_sr, binary=True)
        else:
            array = self.vad_model(y, orig_sr)
        change = False
        if array is not None:
            print("VAD: end of speech detected")
            start_time = time.time()
            if self.client is not None:
                (text_str, audio_output) = self.client(array, orig_sr)
                asr_output_str = ""
                latency_TTS = time.time() - start_time
            else:
                prompt = self.s2t(array)
                if len(prompt.strip().split()) < 2:
                    return (
                        asr_output_str,
                        text_str,
                        audio_output,
                        audio_output1,
                        latency_ASR,
                        latency_LM,
                        latency_TTS,
                        change,
                    )

                asr_output_str = prompt
                start_LM_time = time.time()
                latency_ASR = start_LM_time - start_time
                self.chat.append({"role": self.user_role, "content": prompt})
                chat_messages = self.chat.to_list()
                generated_text = self.LM_pipe(chat_messages)
                start_TTS_time = time.time()
                latency_LM = start_TTS_time - start_LM_time

                self.chat.append({"role": "assistant", "content": generated_text})
                text_str = generated_text
                audio_output = self.text2speech(text_str)
                latency_TTS = time.time() - start_TTS_time
            audio_output1 = (orig_sr, stream)
            stream = y
            change = True
        return (
            asr_output_str,
            text_str,
            audio_output,
            audio_output1,
            latency_ASR,
            latency_LM,
            latency_TTS,
            change,
        )

    def collect_feats():
        return None
