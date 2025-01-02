import argparse
import os
import shutil
import time
from typing import Generator, Optional, Tuple

import gradio as gr
import nltk
import numpy as np
import torch
from huggingface_hub import HfApi
from pyscripts.utils.dialog_eval.ASR_WER import handle_espnet_ASR_WER
from pyscripts.utils.dialog_eval.human_feedback import (
    natural_vote1_last_response,
    natural_vote2_last_response,
    natural_vote3_last_response,
    natural_vote4_last_response,
    relevant_vote1_last_response,
    relevant_vote2_last_response,
    relevant_vote3_last_response,
    relevant_vote4_last_response,
)
from pyscripts.utils.dialog_eval.LLM_Metrics import (
    DialoGPT_perplexity,
    bert_score,
    perplexity,
    vert,
)
from pyscripts.utils.dialog_eval.TTS_intelligibility import (
    handle_espnet_TTS_intelligibility,
)
from pyscripts.utils.dialog_eval.TTS_speech_quality import TTS_psuedomos

from espnet2.sds.espnet_model import ESPnetSDSModelInterface

# ------------------------
# Hyperparameters
# ------------------------

access_token = os.environ.get("HF_TOKEN")
ASR_name = None
LLM_name = None
TTS_name = None
ASR_options = []
LLM_options = []
TTS_options = []
Eval_options = []
upload_to_hub = None
dialogue_model = None

latency_ASR = 0.0
latency_LM = 0.0
latency_TTS = 0.0

text_str = ""
asr_output_str = ""
vad_output = None
audio_output = None
audio_output1 = None
LLM_response_arr = []
total_response_arr = []
callback = gr.CSVLogger()
start_record_time = None
enable_btn = gr.Button(interactive=True, visible=True)

# ------------------------
# Function Definitions
# ------------------------


def parse_args():
    global access_token
    global ASR_name
    global LLM_name
    global TTS_name
    global ASR_options
    global LLM_options
    global TTS_options
    global Eval_options
    global upload_to_hub
    global dialogue_model
    parser = argparse.ArgumentParser(description="Run the app.")
    parser.add_argument(
        "--asr_options",
        required=True,
        help="Provide the possible ASR options available to user.",
    )
    parser.add_argument(
        "--llm_options",
        required=True,
        help="Provide the possible LLM options available to user.",
    )
    parser.add_argument(
        "--tts_options",
        required=True,
        help="Provide the possible TTS options available to user.",
    )
    parser.add_argument(
        "--eval_options",
        required=True,
        help="Provide the possible automatic evaluation metrics available to user.",
    )
    parser.add_argument(
        "--default_asr_model",
        required=False,
        default="pyf98/owsm_ctc_v3.1_1B",
        help="Provide the default ASR model.",
    )
    parser.add_argument(
        "--default_llm_model",
        required=False,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Provide the default LLM model.",
    )
    parser.add_argument(
        "--default_tts_model",
        required=False,
        default="kan-bayashi/ljspeech_vits",
        help="Provide the default TTS model.",
    )
    parser.add_argument(
        "--upload_to_hub",
        required=False,
        default=None,
        help="Hugging Face dataset to upload user data",
    )
    args = parser.parse_args()
    ASR_name = args.default_asr_model
    LLM_name = args.default_llm_model
    TTS_name = args.default_tts_model
    ASR_options = args.asr_options.split(",")
    LLM_options = args.llm_options.split(",")
    TTS_options = args.tts_options.split(",")
    Eval_options = args.eval_options.split(",")
    if ASR_name not in ASR_options:
        print(
            "Changing default ASR model since it is "
            + "not in the possible ASR options"
        )
        ASR_name = ASR_options[0]
    if TTS_name not in TTS_options:
        print(
            "Changing default TTS model since it is "
            + "not in the possible TTS options"
        )
        TTS_name = TTS_options[0]
    if LLM_name not in LLM_options:
        print(
            "Changing default LLM model since it is "
            + "not in the possible LLM options"
        )
        LLM_name = LLM_options[0]
    upload_to_hub = args.upload_to_hub
    dialogue_model = ESPnetSDSModelInterface(
        ASR_name, LLM_name, TTS_name, "Cascaded", access_token
    )


def handle_eval_selection(
    option: str,
    TTS_audio_output: str,
    LLM_Output: str,
    ASR_audio_output: str,
    ASR_transcript: str,
):
    """
    Handles the evaluation of a selected metric based on
    user input and provided outputs.

    This function evaluates different aspects of a
    casacaded conversational AI pipeline, such as:
    Latency, TTS intelligibility, TTS speech quality,
    ASR WER, and text dialog metrics.
    It is designed to integrate with Gradio via
    multiple yield statements,
    allowing updates to be displayed in real time.

    Parameters:
    ----------
    option : str
        The evaluation metric selected by the user.
        Supported options include:
        - "Latency"
        - "TTS Intelligibility"
        - "TTS Speech Quality"
        - "ASR WER"
        - "Text Dialog Metrics"
    TTS_audio_output : np.ndarray
        The audio output generated by the TTS module for evaluation.
    LLM_Output : str
        The text output generated by the LLM module for evaluation.
    ASR_audio_output : np.ndarray
        The audio input/output used for ASR evaluation.
    ASR_transcript : str
        The transcript generated by the ASR module for evaluation.

    Returns:
    -------
    str
        A string representation of the evaluation results.
        The specific result depends on the selected evaluation metric:
        - "Latency": Latencies of ASR, LLM, and TTS modules.
        - "TTS Intelligibility": A range of scores indicating how intelligible
        the TTS audio output is based on different reference ASR models.
        - "TTS Speech Quality": A range of scores representing the
        speech quality of the TTS audio output.
        - "ASR WER": The Word Error Rate (WER) of the ASR output
        based on different judge ASR models.
        - "Text Dialog Metrics": A combination of perplexity,
        diversity metrics, and relevance scores for the dialog.

    Raises:
    ------
    ValueError
        If the `option` parameter does not match any supported evaluation metric.

    Example:
    -------
    >>> result = handle_eval_selection(
            option="Latency",
            TTS_audio_output=audio_array,
            LLM_Output="Generated response",
            ASR_audio_output=audio_input,
            ASR_transcript="Expected transcript"
        )
    >>> print(result)
    "ASR Latency: 0.14
     LLM Latency: 0.42
     TTS Latency: 0.21"
    """
    global LLM_response_arr
    global total_response_arr
    yield (option, gr.Textbox(visible=True))
    if option == "Latency":
        text = (
            f"ASR Latency: {latency_ASR:.2f}\n"
            f"LLM Latency: {latency_LM:.2f}\n"
            f"TTS Latency: {latency_TTS:.2f}"
        )
        yield (None, text)
    elif option == "TTS Intelligibility":
        yield (None, handle_espnet_TTS_intelligibility(TTS_audio_output, LLM_Output))
    elif option == "TTS Speech Quality":
        yield (None, TTS_psuedomos(TTS_audio_output))
    elif option == "ASR WER":
        yield (None, handle_espnet_ASR_WER(ASR_audio_output, ASR_transcript))
    elif option == "Text Dialog Metrics":
        yield (
            None,
            perplexity(LLM_Output.replace("\n", " "))
            + vert(LLM_response_arr)
            + bert_score(total_response_arr)
            + DialoGPT_perplexity(
                ASR_transcript.replace("\n", " "), LLM_Output.replace("\n", " ")
            ),
        )
    elif option is None:
        return
    else:
        raise ValueError(f"Unknown option: {option}")


def handle_eval_selection_E2E(
    option: str,
    TTS_audio_output: str,
    LLM_Output: str,
):
    """
    Handles the evaluation of a selected metric based on user input
    and provided outputs.

    This function evaluates different aspects of an E2E
    conversational AI model, such as:
    Latency, TTS intelligibility, TTS speech quality, and
    text dialog metrics.
    It is designed to integrate with Gradio via
    multiple yield statements,
    allowing updates to be displayed in real time.

    Parameters:
    ----------
    option : str
        The evaluation metric selected by the user.
        Supported options include:
        - "Latency"
        - "TTS Intelligibility"
        - "TTS Speech Quality"
        - "Text Dialog Metrics"
    TTS_audio_output : np.ndarray
        The audio output generated by the TTS module for evaluation.
    LLM_Output : str
        The text output generated by the LLM module for evaluation.

    Returns:
    -------
    str
        A string representation of the evaluation results.
        The specific result depends on the selected evaluation metric:
        - "Latency": Latency of the entire system.
        - "TTS Intelligibility": A range of scores indicating how intelligible the
        TTS audio output is based on different reference ASR models.
        - "TTS Speech Quality": A range of scores representing the
         speech quality of the TTS audio output.
        - "Text Dialog Metrics": A combination of perplexity and
        diversity metrics for the dialog.

    Raises:
    ------
    ValueError
        If the `option` parameter does not match any supported evaluation metric.

    Example:
    -------
    >>> result = handle_eval_selection(
            option="Latency",
            TTS_audio_output=audio_array,
            LLM_Output="Generated response",
        )
    >>> print(result)
    "Total Latency: 2.34"
    """
    global LLM_response_arr
    global total_response_arr
    yield (option, gr.Textbox(visible=True))
    if option == "Latency":
        text = f"Total Latency: {latency_TTS:.2f}"
        yield (None, text)
    elif option == "TTS Intelligibility":
        yield (None, handle_espnet_TTS_intelligibility(TTS_audio_output, LLM_Output))
    elif option == "TTS Speech Quality":
        yield (None, TTS_psuedomos(TTS_audio_output))
    elif option == "Text Dialog Metrics":
        yield (None, perplexity(LLM_Output.replace("\n", " ")) + vert(LLM_response_arr))
    elif option is None:
        return
    else:
        raise ValueError(f"Unknown option: {option}")


def start_warmup():
    """
    Initializes and warms up the dialogue and evaluation model.

    This function is designed to ensure that all
    components of the dialogue model are pre-loaded
    and ready for execution, avoiding delays during runtime.
    """
    global dialogue_model
    global ASR_options
    global LLM_options
    global TTS_options
    global ASR_name
    global LLM_name
    global TTS_name
    for opt_count in range(len(ASR_options)):
        opt = ASR_options[opt_count]
        try:
            for _ in dialogue_model.handle_ASR_selection(opt):
                continue
        except Exception:
            print("Removing " + opt + " from ASR options since it cannot be loaded.")
            ASR_options = ASR_options[:opt_count] + ASR_options[(opt_count + 1) :]
            if opt == ASR_name:
                ASR_name = ASR_options[0]
    for opt_count in range(len(LLM_options)):
        opt = LLM_options[opt_count]
        try:
            for _ in dialogue_model.handle_LLM_selection(opt):
                continue
        except Exception:
            print("Removing " + opt + " from LLM options since it cannot be loaded.")
            LLM_options = LLM_options[:opt_count] + LLM_options[(opt_count + 1) :]
            if opt == LLM_name:
                LLM_name = LLM_options[0]
    for opt_count in range(len(TTS_options)):
        opt = TTS_options[opt_count]
        try:
            for _ in dialogue_model.handle_TTS_selection(opt):
                continue
        except Exception:
            print("Removing " + opt + " from TTS options since it cannot be loaded.")
            TTS_options = TTS_options[:opt_count] + TTS_options[(opt_count + 1) :]
            if opt == TTS_name:
                TTS_name = TTS_options[0]
    dialogue_model.handle_E2E_selection()
    dialogue_model.client = None
    for _ in dialogue_model.handle_TTS_selection(TTS_name):
        continue
    for _ in dialogue_model.handle_ASR_selection(ASR_name):
        continue
    for _ in dialogue_model.handle_LLM_selection(LLM_name):
        continue
    dummy_input = (
        torch.randn(
            (3000),
            dtype=getattr(torch, "float16"),
            device="cpu",
        )
        .cpu()
        .numpy()
    )
    dummy_text = "This is dummy text"
    for opt in Eval_options:
        handle_eval_selection(opt, dummy_input, dummy_text, dummy_input, dummy_text)


def flash_buttons():
    """
    Enables human feedback buttons after displaying system output.
    """
    btn_updates = (enable_btn,) * 8
    yield (
        "",
        "",
    ) + btn_updates


def transcribe(
    stream: np.ndarray,
    new_chunk: Tuple[int, np.ndarray],
    TTS_option: str,
    ASR_option: str,
    LLM_option: str,
    type_option: str,
):
    """
    Processes and transcribes an audio stream in real-time.

    This function handles the transcription of audio input
    and its transformation through a cascaded
    or E2E conversational AI system.
    It dynamically updates the transcription, text generation,
    and synthesized speech output, while managing global states and latencies.

    Args:
        stream: The current audio stream buffer.
            `None` if the stream is being reset (e.g., after user refresh).
        new_chunk: A tuple containing:
            - `sr`: Sample rate of the new audio chunk.
            - `y`: New audio data chunk.
        TTS_option: Selected TTS model option.
        ASR_option: Selected ASR model option.
        LLM_option: Selected LLM model option.
        type_option: Type of system ("Cascaded" or "E2E").

    Yields:
        Tuple[Optional[np.ndarray], Optional[str], Optional[str],
        Optional[Tuple[int, np.ndarray]], Optional[Tuple[int, np.ndarray]]]:
            A tuple containing:
            - Updated stream buffer.
            - ASR output text.
            - Generated LLM output text.
            - Audio output as a tuple of sample rate and audio waveform.
            - User input audio as a tuple of sample rate and audio waveform.

    Notes:
        - Resets the session if the transcription exceeds 5 minutes.
        - Updates the Gradio interface elements dynamically.
        - Manages latencies.
    """
    sr, y = new_chunk
    global text_str
    global chat
    global user_role
    global audio_output
    global audio_output1
    global vad_output
    global asr_output_str
    global start_record_time
    global sids
    global spembs
    global latency_ASR
    global latency_LM
    global latency_TTS
    global LLM_response_arr
    global total_response_arr
    if stream is None:
        # Handle user refresh
        for (
            _,
            _,
            _,
            _,
            asr_output_box,
            text_box,
            audio_box,
            _,
            _,
        ) in dialogue_model.handle_type_selection(
            type_option, TTS_option, ASR_option, LLM_option
        ):
            gr.Info("The models are being reloaded due to a browser refresh.")
            yield (stream, asr_output_box, text_box, audio_box, gr.Audio(visible=False))
        stream = y
        text_str = ""
        audio_output = None
        audio_output1 = None
    else:
        stream = np.concatenate((stream, y))
    (
        asr_output_str,
        text_str,
        audio_output,
        audio_output1,
        latency_ASR,
        latency_LM,
        latency_TTS,
        stream,
        change,
    ) = dialogue_model(
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
    )
    text_str1 = text_str
    if change:
        print("Output changed")
        if asr_output_str != "":
            total_response_arr.append(asr_output_str.replace("\n", " "))
        LLM_response_arr.append(text_str.replace("\n", " "))
        total_response_arr.append(text_str.replace("\n", " "))
    if (text_str != "") and (start_record_time is None):
        start_record_time = time.time()
    elif start_record_time is not None:
        current_record_time = time.time()
        if current_record_time - start_record_time > 300:
            gr.Info(
                "Conversations are limited to 5 minutes. "
                "The session will restart in approximately 60 seconds. "
                "Please wait for the demo to reset. "
                "Close this message once you have read it.",
                duration=None,
            )
            yield stream, gr.Textbox(visible=False), gr.Textbox(
                visible=False
            ), gr.Audio(visible=False), gr.Audio(visible=False)
            if upload_to_hub is not None:
                api.upload_folder(
                    folder_path="flagged_data_points",
                    path_in_repo="checkpoint_" + str(start_record_time),
                    repo_id=upload_to_hub,
                    repo_type="dataset",
                    token=access_token,
                )
            dialogue_model.chat.buffer = []
            text_str = ""
            audio_output = None
            audio_output1 = None
            asr_output_str = ""
            start_record_time = None
            LLM_response_arr = []
            total_response_arr = []
            shutil.rmtree("flagged_data_points")
            os.mkdir("flagged_data_points")
            yield (stream, asr_output_str, text_str1, audio_output, audio_output1)
            yield stream, gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Audio(
                visible=True
            ), gr.Audio(visible=False)

    yield (stream, asr_output_str, text_str1, audio_output, audio_output1)


# ------------------------
# Executable Script
# ------------------------

parse_args()
api = HfApi()
nltk.download("averaged_perceptron_tagger_eng")
start_warmup()
with gr.Blocks(
    title="E2E Spoken Dialog System",
) as demo:
    with gr.Row():
        gr.Markdown(
            """
            ## ESPnet-SDS
            Welcome to our unified web interface for various cascaded and
            E2E spoken dialogue systems built using ESPnet-SDS  toolkit,
            supporting real-time automated evaluation metrics, and
            human-in-the-loop feedback collection.

            For more details on how to use the app, refer to the [README]
            (https://github.com/siddhu001/espnet/tree/sds_demo_recipe/egs2/TEMPLATE/sds1#how-to-use).
        """
        )
    with gr.Row():
        with gr.Column(scale=1):
            user_audio = gr.Audio(
                sources=["microphone"],
                streaming=True,
                waveform_options=gr.WaveformOptions(sample_rate=16000),
            )
            with gr.Row():
                type_radio = gr.Radio(
                    choices=["Cascaded", "E2E"],
                    label="Choose type of Spoken Dialog:",
                    value="Cascaded",
                )
            with gr.Row():
                ASR_radio = gr.Radio(
                    choices=ASR_options,
                    label="Choose ASR:",
                    value=ASR_name,
                )
            with gr.Row():
                LLM_radio = gr.Radio(
                    choices=LLM_options,
                    label="Choose LLM:",
                    value=LLM_name,
                )
            with gr.Row():
                radio = gr.Radio(
                    choices=TTS_options,
                    label="Choose TTS:",
                    value=TTS_name,
                )
            with gr.Row():
                E2Eradio = gr.Radio(
                    choices=["mini-omni"],
                    label="Choose E2E model:",
                    value="mini-omni",
                    visible=False,
                )
            with gr.Row():
                feedback_btn = gr.Button(
                    value=(
                        "Please provide your feedback "
                        "after each system response below."
                    ),
                    visible=True,
                    interactive=False,
                    elem_id="button",
                )
            with gr.Row():
                natural_btn1 = gr.Button(
                    value="Very Natural", visible=False, interactive=False, scale=1
                )
                natural_btn2 = gr.Button(
                    value="Somewhat Awkward", visible=False, interactive=False, scale=1
                )
                natural_btn3 = gr.Button(
                    value="Very Awkward", visible=False, interactive=False, scale=1
                )
                natural_btn4 = gr.Button(
                    value="Unnatural", visible=False, interactive=False, scale=1
                )
            with gr.Row():
                relevant_btn1 = gr.Button(
                    value="Highly Relevant", visible=False, interactive=False, scale=1
                )
                relevant_btn2 = gr.Button(
                    value="Partially Relevant",
                    visible=False,
                    interactive=False,
                    scale=1,
                )
                relevant_btn3 = gr.Button(
                    value="Slightly Irrelevant",
                    visible=False,
                    interactive=False,
                    scale=1,
                )
                relevant_btn4 = gr.Button(
                    value="Completely Irrelevant",
                    visible=False,
                    interactive=False,
                    scale=1,
                )
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Output", autoplay=True, visible=True)
            output_audio1 = gr.Audio(label="Output1", autoplay=False, visible=False)
            output_asr_text = gr.Textbox(label="ASR output")
            output_text = gr.Textbox(label="LLM output")
            eval_radio = gr.Radio(
                choices=[
                    "Latency",
                    "TTS Intelligibility",
                    "TTS Speech Quality",
                    "ASR WER",
                    "Text Dialog Metrics",
                ],
                label="Choose Evaluation metrics:",
            )
            eval_radio_E2E = gr.Radio(
                choices=[
                    "Latency",
                    "TTS Intelligibility",
                    "TTS Speech Quality",
                    "Text Dialog Metrics",
                ],
                label="Choose Evaluation metrics:",
                visible=False,
            )
            output_eval_text = gr.Textbox(label="Evaluation Results")
            state = gr.State()
    with gr.Row():
        privacy_text = gr.Textbox(
            label="Privacy Notice",
            interactive=False,
            value=(
                "By using this demo, you acknowledge that"
                "interactions with this dialog system are collected "
                "for research and improvement purposes. The data "
                "will only be used to enhance the performance and "
                "understanding of the system. If you have any "
                "concerns about data collection, please discontinue "
                "use."
            ),
        )

    btn_list = [
        natural_btn1,
        natural_btn2,
        natural_btn3,
        natural_btn4,
        relevant_btn1,
        relevant_btn2,
        relevant_btn3,
        relevant_btn4,
    ]
    natural_btn_list = [
        natural_btn1,
        natural_btn2,
        natural_btn3,
        natural_btn4,
    ]
    relevant_btn_list = [
        relevant_btn1,
        relevant_btn2,
        relevant_btn3,
        relevant_btn4,
    ]
    natural_response = gr.Textbox(
        label="natural_response", visible=False, interactive=False
    )
    diversity_response = gr.Textbox(
        label="diversity_response", visible=False, interactive=False
    )
    ip_address = gr.Textbox(label="ip_address", visible=False, interactive=False)
    callback.setup(
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        "flagged_data_points",
    )
    user_audio.stream(
        transcribe,
        inputs=[state, user_audio, radio, ASR_radio, LLM_radio, type_radio],
        outputs=[state, output_asr_text, output_text, output_audio, output_audio1],
    ).then(
        lambda *args: callback.flag(list(args)), [user_audio], None, preprocess=False
    )
    radio.change(
        fn=dialogue_model.handle_TTS_selection,
        inputs=[radio],
        outputs=[output_asr_text, output_text, output_audio],
    )
    LLM_radio.change(
        fn=dialogue_model.handle_LLM_selection,
        inputs=[LLM_radio],
        outputs=[output_asr_text, output_text, output_audio],
    )
    ASR_radio.change(
        fn=dialogue_model.handle_ASR_selection,
        inputs=[ASR_radio],
        outputs=[output_asr_text, output_text, output_audio],
    )
    eval_radio.change(
        fn=handle_eval_selection,
        inputs=[eval_radio, output_audio, output_text, output_audio1, output_asr_text],
        outputs=[eval_radio, output_eval_text],
    )
    eval_radio_E2E.change(
        fn=handle_eval_selection_E2E,
        inputs=[eval_radio_E2E, output_audio, output_text],
        outputs=[eval_radio_E2E, output_eval_text],
    )
    type_radio.change(
        fn=dialogue_model.handle_type_selection,
        inputs=[type_radio, radio, ASR_radio, LLM_radio],
        outputs=[
            radio,
            ASR_radio,
            LLM_radio,
            E2Eradio,
            output_asr_text,
            output_text,
            output_audio,
            eval_radio,
            eval_radio_E2E,
        ],
    )
    output_audio.play(
        flash_buttons, [], [natural_response, diversity_response] + btn_list
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
        ],
        None,
        preprocess=False,
    )
    natural_btn1.click(
        natural_vote1_last_response,
        [],
        [natural_response, ip_address] + natural_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    natural_btn2.click(
        natural_vote2_last_response,
        [],
        [natural_response, ip_address] + natural_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    natural_btn3.click(
        natural_vote3_last_response,
        [],
        [natural_response, ip_address] + natural_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    natural_btn4.click(
        natural_vote4_last_response,
        [],
        [natural_response, ip_address] + natural_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    relevant_btn1.click(
        relevant_vote1_last_response,
        [],
        [diversity_response, ip_address] + relevant_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    relevant_btn2.click(
        relevant_vote2_last_response,
        [],
        [diversity_response, ip_address] + relevant_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    relevant_btn3.click(
        relevant_vote3_last_response,
        [],
        [diversity_response, ip_address] + relevant_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
    relevant_btn4.click(
        relevant_vote4_last_response,
        [],
        [diversity_response, ip_address] + relevant_btn_list,
    ).then(
        lambda *args: callback.flag(list(args)),
        [
            user_audio,
            output_asr_text,
            output_text,
            output_audio,
            output_audio1,
            type_radio,
            ASR_radio,
            LLM_radio,
            radio,
            E2Eradio,
            natural_response,
            diversity_response,
            ip_address,
        ],
        None,
        preprocess=False,
    )
demo.launch(share=True)
