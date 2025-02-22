import time
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
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


try:
    import gradio as gr

    is_gradio_available = True
except ImportError:
    is_gradio_available = False


class ESPnetSDSModelInterface(AbsESPnetModel):
    """Web Interface for Spoken Dialog System models

    This class provides a unified interface to integrate ASR, TTS, and LLM modules
    for cascaded spoken dialog systems as well as also
    supports E2E spoken dialog systems.
    It supports real-time interactions,
    including VAD (Voice Activity Detection) based conversation management.
    """

    @typechecked
    def __init__(
        self,
        ASR_option: str,
        LLM_option: str,
        TTS_option: str,
        type_option: str,
        access_token: str,
    ):
        """Initializer method.

        Args:
            ASR_option (str):
                The selected ASR model option to use for speech-to-text
                processing.
            LLM_option (str):
                The selected LLM model option for generating text responses.
            TTS_option (str):
                The selected TTS model option for text-to-speech synthesis.
            type_option (str):
                The type of SDS interaction to perform (e.g., cascaded or E2E).
            access_token (str):
                The access token for accessing models hosted on Hugging Face.
        """
        if not is_gradio_available:
            raise ImportError("Error: Gradio is not properly installed.")
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
                "content": (
                    "You are a helpful and friendly AI "
                    "assistant. "
                    "You are polite, respectful, and aim to "
                    "provide concise and complete responses of "
                    "less than 15 words."
                ),
            }
        )
        self.user_role = "user"

    def handle_TTS_selection(self, option: str):
        """Handles the selection and initialization of a Text-to-Speech (TTS) model.

        This method dynamically loads the selected TTS model based
        on the provided option.
        If the selected model is already active, it avoids reloading to save resources.
        The method temporarily removes the visibility of Gradio outputs during the
        initialization process to indicate progress.

        Args:
            option (str):
                The name of the TTS model to load.
        """
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

    def handle_LLM_selection(self, option: str):
        """Handles the selection and initialization of a LLM.

        This method dynamically loads the selected LLM based
        on the provided option.
        If the selected model is already active, it avoids reloading to save resources.
        The method temporarily removes the visibility of Gradio outputs during the
        initialization process to indicate progress.

        Args:
            option (str):
                The name of the LLM to load.
        """
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

    def handle_ASR_selection(self, option: str):
        """Handles the selection and initialization of ASR model.

        This method dynamically loads the selected ASR based
        on the provided option.
        If the selected model is already active, it avoids reloading to save resources.
        The method temporarily removes the visibility of Gradio outputs during the
        initialization process to indicate progress.

        Args:
            option (str):
                The name of the ASR to load.
        """
        if option == "librispeech_asr":
            option = (
                "espnet/"
                "simpleoier_librispeech_asr_train_asr_conformer7"
                "_wavlm_large_raw_en_bpe5000_sp"
            )
        if self.ASR_curr_name is not None:
            if option == self.ASR_curr_name:
                return
        yield gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Audio(
            visible=False
        )
        self.ASR_curr_name = option
        if option == "espnet/owsm_v3.1_ebf":
            self.s2t = OWSMModel()
        elif "owsm_ctc" in option:
            self.s2t = OWSMCTCModel(tag=option)
        elif "whisper" in option:
            self.s2t = WhisperASRModel(tag=option.replace("whisper-", ""))
        else:
            self.s2t = ESPnetASRModel(tag=option)

        self.s2t.warmup()
        yield gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Audio(visible=True)

    def handle_E2E_selection(self):
        """Handles the selection and initialization of E2E model Mini-Omni.

        This method dynamically loads the E2E spoken dialog model.
        If the model is already active, it avoids reloading to save resources.
        """
        if self.client is None:
            self.client = MiniOmniE2EModel()
            self.client.warmup()

    def handle_type_selection(
        self, option: str, TTS_radio: str, ASR_radio: str, LLM_radio: str
    ):
        """Handles the selection of the spoken dialogue model type (Cascaded or E2E)

        and dynamically updates the interface based on the selected option.

        This method manages the initialization of ASR, TTS, and LLM models
        for Cascaded systems or switches to an End-to-End system.
        The Gradio interface components are updated accordingly.

        Args:
            option (str):
                The selected spoken dialogue system.
            TTS_radio (str):
                The selected TTS model for the Cascaded system.
            ASR_radio (str):
                The selected ASR model for the Cascaded system.
            LLM_radio (str):
                The selected LLM model for the Cascaded system.
        """
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
            self.type_option = "Cascaded"
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
            self.type_option = "E2E"
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
        y: np.ndarray,
        sr: int,
        stream: np.ndarray,
        asr_output_str: Optional[str],
        text_str: Optional[str],
        audio_output: Optional[Tuple[int, np.ndarray]],
        audio_output1: Optional[Tuple[int, np.ndarray]],
        latency_ASR: float,
        latency_LM: float,
        latency_TTS: float,
    ):
        """Processes audio input to generate ASR, LLM, and TTS outputs

        while calculating latencies.

        This method handles both Cascaded and End-to-End setups.

        Args:
            y: Input audio array.
            sr : Sampling rate of the input audio.
            stream : The current audio stream buffer.
            asr_output_str : Previously generated ASR output string.
            text_str : Previously generated LLM text response.
            audio_output : Previously generated TTS audio output.
            audio_output1 (): Placeholder for audio stream.
            latency_ASR (float): Latency for ASR processing.
            latency_LM (float): Latency for LLM processing.
            latency_TTS (float): Latency for TTS processing.

        Returns:
            Tuple[str, str, Optional[Tuple[int, np.ndarray]],
            Optional[Tuple[int, np.ndarray]], float, float, float, bool]:
                - Updated ASR output string.
                - Updated LLM-generated text.
                - Updated TTS audio output.
                - Updated user audio stream output.
                - ASR latency.
                - LLM latency.
                - TTS latency.
                - Update audio stream
                - Change flag indicating if output was updated.
        """
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
                        stream,
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
            stream,
            change,
        )

    def collect_feats():
        return None
