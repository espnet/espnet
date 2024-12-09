import os
import shutil
from espnet2.sds.asr.espnet_asr import ESPnetASRModel
from espnet2.sds.asr.owsm_asr import OWSMModel
from espnet2.sds.asr.owsm_ctc_asr import OWSMCTCModel
from espnet2.sds.asr.whisper_asr import WhisperASRModel
from espnet2.sds.tts.espnet_tts import ESPnetTTSModel
from espnet2.sds.tts.chat_tts import ChatTTSModel
from espnet2.sds.llm.hugging_face_llm import HuggingFaceLLM
from espnet2.sds.vad.webrtc_vad import WebrtcVADModel
from espnet2.sds.eval.TTS_intelligibility import handle_espnet_TTS_intelligibility
from espnet2.sds.eval.ASR_WER import handle_espnet_ASR_WER
from espnet2.sds.eval.TTS_speech_quality import TTS_psuedomos
from espnet2.sds.eval.LLM_Metrics import perplexity, vert, bert_score, DialoGPT_perplexity
from espnet2.sds.utils.chat import Chat
import argparse

access_token = os.environ.get("HF_TOKEN")
ASR_name=None
LLM_name=None
TTS_name=None
ASR_options=[]
LLM_options=[]
TTS_options=[]
Eval_options=[]
upload_to_hub=None
def read_args():
    global access_token
    global ASR_name
    global LLM_name
    global TTS_name
    global ASR_options
    global LLM_options
    global TTS_options
    global Eval_options
    global upload_to_hub
    parser = argparse.ArgumentParser(description="Run the app with HF_TOKEN as a command-line argument.")
    parser.add_argument("--HF_TOKEN", required=True, help="Provide the Hugging Face token.")
    parser.add_argument("--asr_options", required=True, help="Provide the possible ASR options available to user.")
    parser.add_argument("--llm_options", required=True, help="Provide the possible LLM options available to user.")
    parser.add_argument("--tts_options", required=True, help="Provide the possible TTS options available to user.")
    parser.add_argument("--eval_options", required=True, help="Provide the possible automatic evaluation metrics available to user.")
    parser.add_argument("--default_asr_model", required=False, default="pyf98/owsm_ctc_v3.1_1B", help="Provide the default ASR model.")
    parser.add_argument("--default_llm_model", required=False, default="meta-llama/Llama-3.2-1B-Instruct", help="Provide the default ASR model.")
    parser.add_argument("--default_tts_model", required=False, default="kan-bayashi/ljspeech_vits", help="Provide the default ASR model.")
    parser.add_argument("--upload_to_hub", required=False, default=None, help="Hugging Face dataset to upload user data")
    args = parser.parse_args()
    access_token=args.HF_TOKEN
    ASR_name=args.default_asr_model
    LLM_name=args.default_llm_model
    TTS_name=args.default_tts_model
    ASR_options=args.asr_options.split(",")
    LLM_options=args.llm_options.split(",")
    TTS_options=args.tts_options.split(",")
    Eval_options=args.eval_options.split(",")
    upload_to_hub=args.upload_to_hub

read_args()
from huggingface_hub import HfApi

api = HfApi()
import nltk
nltk.download('averaged_perceptron_tagger_eng')
import gradio as gr


import numpy as np

chat = Chat(2)
chat.init_chat({"role": "system", "content": "You are a helpful and friendly AI assistant. The user is talking to you with their voice and you should respond in a conversational style. You are polite, respectful, and aim to provide concise and complete responses of less than 15 words."})
user_role = "user"

text2speech=None
s2t=None
LM_pipe=None

latency_ASR=0.0
latency_LM=0.0
latency_TTS=0.0

text_str=""
asr_output_str=""
vad_output=None
audio_output = None
audio_output1 = None
LLM_response_arr=[]
total_response_arr=[]

def handle_selection(option):
    yield gr.Textbox(visible=False),gr.Textbox(visible=False),gr.Audio(visible=False)
    global text2speech
    tag = option 
    if tag=="ChatTTS":
        text2speech = ChatTTSModel()
    else:
        text2speech = ESPnetTTSModel(tag)
    text2speech.warmup()
    yield gr.Textbox(visible=True),gr.Textbox(visible=True),gr.Audio(visible=True)

def handle_LLM_selection(option):
    yield gr.Textbox(visible=False),gr.Textbox(visible=False),gr.Audio(visible=False)
    global LM_pipe
    LM_pipe = HuggingFaceLLM(access_token=access_token,tag = option)
    LM_pipe.warmup()
    yield gr.Textbox(visible=True),gr.Textbox(visible=True),gr.Audio(visible=True)

def handle_ASR_selection(option):
    yield gr.Textbox(visible=False),gr.Textbox(visible=False),gr.Audio(visible=False)
    if option=="librispeech_asr":
        option="espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp"
    global s2t
    if option=="espnet/owsm_v3.1_ebf":
        s2t = OWSMModel()
    elif option=="espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp":
        s2t = ESPnetASRModel(tag=option)
    elif option=="whisper":
        s2t = WhisperASRModel()
    else:
        s2t = OWSMCTCModel(tag=option)

    s2t.warmup()
    yield gr.Textbox(visible=True),gr.Textbox(visible=True),gr.Audio(visible=True)

def handle_eval_selection(option, TTS_audio_output, LLM_Output, ASR_audio_output, ASR_transcript):
    global LLM_response_arr
    global total_response_arr
    yield (option,gr.Textbox(visible=True))
    if option=="Latency":
        text=f"ASR Latency: {latency_ASR:.2f}\nLLM Latency: {latency_LM:.2f}\nTTS Latency: {latency_TTS:.2f}"
        yield (None,text)
    elif option=="TTS Intelligibility":
        yield (None,handle_espnet_TTS_intelligibility(TTS_audio_output,LLM_Output))
    elif option=="TTS Speech Quality":
        yield (None,TTS_psuedomos(TTS_audio_output))
    elif option=="ASR WER":
        yield (None,handle_espnet_ASR_WER(ASR_audio_output, ASR_transcript))
    elif option=="Text Dialog Metrics":
        yield (None,perplexity(LLM_Output.replace("\n"," "))+vert(LLM_response_arr)+bert_score(total_response_arr)+DialoGPT_perplexity(ASR_transcript.replace("\n"," "),LLM_Output.replace("\n"," ")))

for _ in handle_selection(TTS_name):
    continue
for _ in handle_ASR_selection(ASR_name):
    continue
for _ in handle_LLM_selection(LLM_name):
    continue
vad_model=WebrtcVADModel()

callback = gr.CSVLogger()
start_record_time=None
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False, visible=False)
def flash_buttons():
    btn_updates = (enable_btn,) * 8
    print(enable_btn)
    yield ("","",)+btn_updates


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    return ip


def vote_last_response(vote_type, request: gr.Request):
    with open("save_dict.json", "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


def natural_vote1_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Very Natural (voted). ip: {ip_address1}")
    return ("Very Natural",ip_address1,)+(disable_btn,) * 4

def natural_vote2_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Somewhat Awkward (voted). ip: {ip_address1}")
    return ("Somewhat Awkward",ip_address1,)+(disable_btn,) * 4

def natural_vote3_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Very Awkward (voted). ip: {ip_address1}")
    return ("Very Awkward",ip_address1,)+(disable_btn,) * 4

def natural_vote4_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Unnatural (voted). ip: {ip_address1}")
    return ("Unnatural",ip_address1,)+(disable_btn,) * 4

def relevant_vote1_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Highly Relevant (voted). ip: {ip_address1}")
    return ("Highly Relevant",ip_address1,)+(disable_btn,) * 4

def relevant_vote2_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Partially Relevant (voted). ip: {ip_address1}")
    return ("Partially Relevant",ip_address1,)+(disable_btn,) * 4

def relevant_vote3_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Slightly Irrelevant (voted). ip: {ip_address1}")
    return ("Slightly Irrelevant",ip_address1,)+(disable_btn,) * 4

def relevant_vote4_last_response(
    request: gr.Request
):
    ip_address1=get_ip(request)
    print(f"Completely Irrelevant (voted). ip: {ip_address1}")
    return ("Completely Irrelevant",ip_address1,)+(disable_btn,) * 4

import json
import time

def transcribe(stream, new_chunk, option, asr_option):
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
        stream=y
        chat.init_chat({"role": "system", "content": "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise and complete responses of less than 15 words."})
        text_str=""
        audio_output = None
        audio_output1 = None
    else:
        stream=np.concatenate((stream,y))
    orig_sr=sr
    sr=16000
    array=vad_model(y,orig_sr)
    
    if array is not None:
        print("VAD: end of speech detected")
        start_time = time.time()
        prompt=s2t(array)
        if len(prompt.strip().split())<2:
            text_str1=text_str    
            yield (stream, asr_output_str, text_str1, audio_output, audio_output1)
            return
        
        
        asr_output_str=prompt
        total_response_arr.append(prompt.replace("\n"," "))
        start_LM_time=time.time()
        latency_ASR=(start_LM_time - start_time)
        chat.append({"role": user_role, "content": prompt})
        chat_messages = chat.to_list()
        generated_text = LM_pipe(chat_messages)
        start_TTS_time=time.time()
        latency_LM=(start_TTS_time - start_LM_time)

        chat.append({"role": "assistant", "content": generated_text})
        text_str=generated_text
        LLM_response_arr.append(text_str.replace("\n"," "))
        total_response_arr.append(text_str.replace("\n"," "))
        audio_output=text2speech(text_str)
        audio_output1=(orig_sr,stream)
        stream=y
        latency_TTS=(time.time() - start_TTS_time)
    text_str1=text_str
    if ((text_str!="") and (start_record_time is None)):
        start_record_time=time.time()
    elif start_record_time is not None:
        current_record_time=time.time()
        if current_record_time-start_record_time>300:
            gr.Info("Conversations are limited to 5 minutes. The session will restart in approximately 60 seconds. Please wait for the demo to reset. Close this message once you have read it.", duration=None)
            yield stream,gr.Textbox(visible=False),gr.Textbox(visible=False),gr.Audio(visible=False),gr.Audio(visible=False)
            if upload_to_hub is not None:
                api.upload_folder(
                    folder_path="flagged_data_points",
                    path_in_repo="checkpoint_"+str(start_record_time),
                    repo_id=upload_to_hub,
                    repo_type="dataset",
                    token=access_token,
                )
            chat.buffer=[{"role": "system", "content": "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise and complete responses of less than 15 words."}]
            text_str=""
            audio_output = None
            audio_output1 = None
            asr_output_str = ""
            start_record_time = None
            LLM_response_arr=[]
            total_response_arr=[]
            shutil.rmtree('flagged_data_points')
            os.mkdir("flagged_data_points")
            yield (stream,asr_output_str,text_str1, audio_output, audio_output1)
            yield stream,gr.Textbox(visible=True),gr.Textbox(visible=True),gr.Audio(visible=True),gr.Audio(visible=False)
    
    yield (stream,asr_output_str,text_str1, audio_output, audio_output1)


with gr.Blocks(
        title="E2E Spoken Dialog System",
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                user_audio = gr.Audio(sources=["microphone"], streaming=True, waveform_options=gr.WaveformOptions(sample_rate=16000))
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
                    feedback_btn = gr.Button(
                        value="Please provide your feedback after each system response below.", visible=True, interactive=False, elem_id="button"
                    )
                with gr.Row():
                    natural_btn1 = gr.Button(
                        value="Very Natural", visible=False, interactive=False, scale=1
                    )
                    natural_btn2 = gr.Button(
                        value="Somewhat Awkward", visible=False, interactive=False, scale=1
                    )
                    natural_btn3 = gr.Button(value="Very Awkward", visible=False, interactive=False, scale=1)
                    natural_btn4 = gr.Button(
                        value="Unnatural", visible=False, interactive=False, scale=1
                    )
                with gr.Row():
                    relevant_btn1 = gr.Button(
                        value="Highly Relevant", visible=False, interactive=False, scale=1
                    )
                    relevant_btn2 = gr.Button(
                        value="Partially Relevant", visible=False, interactive=False, scale=1
                    )
                    relevant_btn3 = gr.Button(value="Slightly Irrelevant", visible=False, interactive=False, scale=1)
                    relevant_btn4 = gr.Button(
                        value= "Completely Irrelevant", visible=False, interactive=False, scale=1
                    )
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Output", autoplay=True, visible=True)
                output_audio1 = gr.Audio(label="Output1", autoplay=False, visible=False)
                output_asr_text = gr.Textbox(label="ASR output")
                output_text = gr.Textbox(label="LLM output")
                eval_radio = gr.Radio(
                    choices=["Latency", "TTS Intelligibility", "TTS Speech Quality", "ASR WER","Text Dialog Metrics"],
                    label="Choose Evaluation metrics:",
                )
                output_eval_text = gr.Textbox(label="Evaluation Results")
                state = gr.State()
        with gr.Row():
            privacy_text = gr.Textbox(label="Privacy Notice",interactive=False, value="By using this demo, you acknowledge that interactions with this dialog system are collected for research and improvement purposes. The data will only be used to enhance the performance and understanding of the system. If you have any concerns about data collection, please discontinue use.")
        
        btn_list=[
                natural_btn1,
                natural_btn2,
                natural_btn3,
                natural_btn4,
                relevant_btn1,
                relevant_btn2,
                relevant_btn3,
                relevant_btn4,
        ]
        natural_btn_list=[
            natural_btn1,
            natural_btn2,
            natural_btn3,
            natural_btn4,
        ]
        relevant_btn_list=[
            relevant_btn1,
            relevant_btn2,
            relevant_btn3,
            relevant_btn4,
        ]
        natural_response = gr.Textbox(label="natural_response",visible=False,interactive=False)
        diversity_response = gr.Textbox(label="diversity_response",visible=False,interactive=False)
        ip_address = gr.Textbox(label="ip_address",visible=False,interactive=False)
        callback.setup([user_audio, output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address],"flagged_data_points")
        user_audio.stream(transcribe, inputs=[state, user_audio, radio, ASR_radio], outputs=[state, output_asr_text, output_text, output_audio, output_audio1]).then(lambda *args: callback.flag(list(args)),[user_audio], None,preprocess=False)
        radio.change(fn=handle_selection, inputs=[radio], outputs=[output_asr_text, output_text, output_audio])
        LLM_radio.change(fn=handle_LLM_selection, inputs=[LLM_radio], outputs=[output_asr_text, output_text, output_audio])
        ASR_radio.change(fn=handle_ASR_selection, inputs=[ASR_radio], outputs=[output_asr_text, output_text, output_audio])
        eval_radio.change(fn=handle_eval_selection, inputs=[eval_radio,output_audio,output_text,output_audio1,output_asr_text], outputs=[eval_radio,output_eval_text])
        output_audio.play(
            flash_buttons, [], [natural_response,diversity_response]+btn_list
        ).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio], None,preprocess=False)
        natural_btn1.click(natural_vote1_last_response,[],[natural_response,ip_address]+natural_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        natural_btn2.click(natural_vote2_last_response,[],[natural_response,ip_address]+natural_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        natural_btn3.click(natural_vote3_last_response,[],[natural_response,ip_address]+natural_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        natural_btn4.click(natural_vote4_last_response,[],[natural_response,ip_address]+natural_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        relevant_btn1.click(relevant_vote1_last_response,[],[diversity_response,ip_address]+relevant_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        relevant_btn2.click(relevant_vote2_last_response,[],[diversity_response,ip_address]+relevant_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        relevant_btn3.click(relevant_vote3_last_response,[],[diversity_response,ip_address]+relevant_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)
        relevant_btn4.click(relevant_vote4_last_response,[],[diversity_response,ip_address]+relevant_btn_list).then(lambda *args: callback.flag(list(args)),[user_audio,output_asr_text, output_text, output_audio,output_audio1,ASR_radio,LLM_radio,radio,natural_response,diversity_response,ip_address], None,preprocess=False)

demo.launch(share=True)
