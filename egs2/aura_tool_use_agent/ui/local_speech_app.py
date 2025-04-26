import os
import sys
import time
import numpy as np
import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from collections import deque
import datetime
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import re
import nltk
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from agent.actions.chat_action import ChatAction
# Now import the agent module
from agent.controller.controller import Controller

import os

# Set custom temp directory before importing gradio
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.path.expanduser("~"), "gradio_temp")
# Make sure the directory exists
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)


try:
    nltk_resources = ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
    for resource in nltk_resources:
        try:
            nltk.data.find(f'taggers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
except Exception as e:
    print(f"Failed to download NLTK resources: {e}")

latency_ASR = 0.0
latency_LLM = 0.0
latency_TTS = 0.0
conversation_history = []

HF_KEY = os.getenv('HF_API_KEY')

# Improved device handling - consistent variable for all components
if torch.cuda.is_available():
    device = "cuda"
    torch_device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    torch_device = torch.device("cpu")
    print("CUDA is not available. Using CPU for processing.")

ASR_OPTIONS = {
    "OWSM CTC v3.1 1B": "espnet/owsm_ctc_v3.1_1B",
    "Whisper Tiny": "openai/whisper-tiny",
    "OWSM v3.2": "espnet/owsm_v3.2",
    "OWSM v3.1 EBF Small": "espnet/owsm_v3.1_ebf_small",
    # "Wav2Vec2 Small": "facebook/wav2vec2-base-960h",
    # "ESPnet English": "espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan",
    
    # "OWSM CTC v3.2 1B": "espnet/owsm_ctc_v3.2_ft_1B"
}

# ESPnet TTS options
ESPNET_TTS_OPTIONS = {
    "ESPnet LJSpeech VITS": "kan-bayashi/ljspeech_vits", 
    "ESPnet LJSpeech FastSpeech2": "kan-bayashi/ljspeech_fastspeech2"
}

LLM_OPTIONS = {
    "Agent": "llama-70b",
    # "Llama Fine-tuned": "llama-ft",
    # Add other options here
}

# Cache for loaded models
asr_models = {}
llm_models = {}
espnet_models = {}
tts_models = {}
owsm_models = {}

# ======================
# Initialize Agent Controller
# ======================

controller = Controller()

# ======================
# New Agent Functions
# ======================
def get_user_input(audio_input, asr_model_name):
    """
    Get user input via recording and transcription
    
    Args:
        audio_input: Audio input data from Gradio
        asr_model_name: The ASR model to use for transcription
        
    Returns:
        str: The transcribed text from the user's speech
    """
    if audio_input is None:
        print("No audio input provided")
        return ""
    
    try:
        sr, audio_data = audio_input
        print(f"Processing user input - Audio: SR={sr}, shape={audio_data.shape}")
        
        # Audio data is already loaded as numpy array, no need to save to disk
        # We can pass it directly to the transcribe_audio function
        transcript = transcribe_audio(audio_data, sr, asr_model_name)
        print(f"User input transcribed: {transcript}")
        
        return transcript
    except Exception as e:
        print(f"Error in get_user_input: {e}")
        return ""

        
def agent_output(response_text, tts_model_name=None):
    """
    Generate speech output for agent response
    
    Args:
        response_text: Text response from the agent
        tts_model_name: The TTS model to use for synthesis
        
    Returns:
        tuple: Audio output data (sample_rate, audio_data)
    """
    print(f"Generating speech for agent response: {response_text}")
    
    # Synthesize speech from the response text
    audio_output = synthesize_speech(response_text, tts_model_name)
    
    # Handle TTS failure
    if audio_output is None:
        print("Failed to synthesize speech, returning error message")
        
    return audio_output

def update_conversation_display(user_text=None, agent_text=None):
    """
    Update the conversation history and display
    
    Args:
        user_text: Text from the user (if any)
        agent_text: Text from the agent (if any)
        
    Returns:
        str: HTML for displaying the updated conversation
    """
    global conversation_history
    
    # Function can be called in two ways:
    # 1. With specific texts to add to the conversation
    # 2. Without arguments, just to refresh the display
    
    # Add user message if provided
    if user_text and user_text.strip():
        print(f"Adding user message to conversation: {user_text}")
        conversation_history.append({"role": "user", "content": user_text})
    
    # Add agent message if provided
    if agent_text and agent_text.strip():
        print(f"Adding agent message to conversation: {agent_text}")
        conversation_history.append({"role": "assistant", "content": agent_text})
    
    # Generate and return the HTML display
    return display_conversation()

# ======================
# ======================

def synthesize_speech(text, tts_model_name=None):
    """Synthesize speech from text using the selected TTS model"""
    global latency_TTS
    
    start_time = time.time()
    print(f"Starting TTS synthesis with model: {tts_model_name}")
    
    audio_output = None
    
    try:
        if not tts_model_name or tts_model_name not in ESPNET_TTS_OPTIONS:
            print("Invalid or missing TTS model name")
            return None
            
        model = load_tts_model(tts_model_name)
        if model is None:
            print("Failed to load TTS model")
            return None
            
        # Check if text is valid
        if not text or len(text.strip()) == 0:
            print("Empty text input provided to TTS, skipping synthesis")
            return None
        
        # Pre-process the text to ensure it's valid for the model
        cleaned_text = re.sub(r'[^\w\s.,!?;:\-\'"]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if len(cleaned_text) == 0:
            print("Text contains only special characters, using fallback text")
            cleaned_text = "I'm sorry, I couldn't process that text."
        
        print(f"Processing TTS with text: '{cleaned_text}'")
        
        # Try synthesis with different text lengths if needed
        try:
            output = model(cleaned_text)
        except Exception as e:
            print(f"TTS attempt failed: {e}")
            
            # If text is too long, try with shorter text
            if len(cleaned_text) > 100:
                print("Trying with shorter text")
                shortened_text = cleaned_text[:100] + "..."
                try:
                    output = model(shortened_text)
                except Exception:
                    print("Shortened text attempt failed, using fallback")
                    output = model("I'm sorry, I couldn't synthesize the response.")
            else:
                # If text is already short, use fallback
                print("Using fallback text for TTS")
                output = model("I'm sorry, I couldn't synthesize the response.")
        
        # Extract audio data
        sr = model.fs
        audio_data = output["wav"].cpu().numpy()
        audio_output = (sr, audio_data)
        print(f"Successfully synthesized speech with {tts_model_name}")
            
    except Exception as e:
        print(f"Error in TTS synthesis: {e}")
        # No need for additional fallback here since we already have fallbacks above
    
    latency_TTS = time.time() - start_time
    print(f"TTS synthesis completed in {latency_TTS:.2f} seconds")
    return audio_output

def load_tts_model(model_name):
    """Load TTS model with error handling"""
    global tts_models
    
    if model_name not in tts_models:
        print(f"Loading TTS model: {model_name}")
        
        if model_name in ESPNET_TTS_OPTIONS:
            model_id = ESPNET_TTS_OPTIONS[model_name]
            try:
                # Use the global device variable
                tts_models[model_name] = Text2Speech.from_pretrained(model_id, device=device)
                print(f"Successfully loaded ESPnet TTS model: {model_name}")
            except Exception as e:
                print(f"Error loading ESPnet TTS model {model_name}: {e}")
                tts_models[model_name] = None
    
    return tts_models[model_name]

def load_asr_model(model_name):
    """Load ASR model with error handling"""
    global asr_models, espnet_models, owsm_models
    
    if model_name not in asr_models and model_name not in espnet_models and model_name not in owsm_models:
        print(f"Loading ASR model: {model_name}")
        
        if "OWSM" in model_name:
            model_id = ASR_OPTIONS[model_name]
            try:
                d = ModelDownloader(cachedir = "/data/user_data/gganeshl/speech")
                model_dict = d.download_and_unpack(model_id)
                
                if "train_config" in model_dict:
                    model_dict.pop("train_config")
                owsm_models[model_name] = Speech2TextGreedySearch(
                    **model_dict,
                    device=device,  # Use global device
                    use_flash_attn=torch.cuda.is_available(),  # Only use if CUDA is available
                    lang_sym='<eng>',
                    task_sym='<asr>',
                )
                print(f"Successfully loaded OWSM ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading OWSM ASR model {model_name}: {e}")
                owsm_models[model_name] = None
                
            return owsm_models[model_name]
            
        elif "ESPnet" in model_name and "OWSM" not in model_name:
            model_id = ASR_OPTIONS[model_name]
            try:
                d = ModelDownloader(cachedir = "/data/user_data/gganeshl/speech")
                model_dict = d.download_and_unpack(model_id)
                
                if "train_config" in model_dict:
                    model_dict.pop("train_config")
                
                speech2text = Speech2Text(
                    **model_dict,
                    device=device,  # Use global device
                    minlenratio=0.0,
                    maxlenratio=0.0,
                    ctc_weight=0.3,
                    beam_size=10,
                    batch_size=0,
                    nbest=1,
                )
                espnet_models[model_name] = speech2text
                print(f"Successfully loaded ESPnet ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading ESPnet ASR model {model_name}: {e}")
                espnet_models[model_name] = None
            
            return espnet_models[model_name]
        else:
            model_id = ASR_OPTIONS[model_name]
            try:
                asr_models[model_name] = pipeline(
                    "automatic-speech-recognition", 
                    model=model_id,
                    device=0 if device == "cuda" else -1  # Use 0 for CUDA, -1 for CPU
                )
                print(f"Successfully loaded ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading ASR model {model_name}: {e}")
                asr_models[model_name] = None
    
    if model_name in owsm_models:
        return owsm_models[model_name]
    elif model_name in espnet_models:
        return espnet_models[model_name]
    else:
        return asr_models[model_name]

def load_llm_model(model_name):
    """Load LLM model with error handling"""
    global llm_models
    
    if model_name not in llm_models:
        print(f"Loading LLM model: {model_name}")
        model_id = LLM_OPTIONS[model_name]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_KEY)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map='auto' if device == "cuda" else None,  # Use device_map for CUDA
                token=HF_KEY
            )
            
            llm_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            print(f"Successfully loaded LLM model: {model_name}")
        except Exception as e:
            print(f"Error loading LLM model {model_name}: {e}")
            llm_models[model_name] = None
    
    return llm_models[model_name]

def transcribe_audio(audio_data, sr, asr_model_name):
    """Transcribe audio using selected ASR model"""
    global latency_ASR
    
    start_time = time.time()
    
    try:
        model = load_asr_model(asr_model_name)
        
        if model is None:
            transcript = "ASR model unavailable. Please try a different model."
        else:
            # Convert audio to float32 for compatibility with ASR models
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if audio_data.max() > 1.0:
                    audio_data = audio_data / 32768.0  # Normalize from int16 range
            
            # Check if it's an OWSM model 
            if "OWSM" in asr_model_name:
                # Ensure audio is at the right sample rate
                if sr != 16000:
                    # Resample audio to 16kHz if needed
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                # Try direct call method
                result = model(audio_data)
                
                # Process and clean the result - remove <eng><asr> tags
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], tuple) and len(result[0]) > 0:
                        transcript = result[0][0]  # Get first hypothesis
                    else:
                        transcript = str(result[0])
                else:
                    transcript = str(result)
                
                # Clean OWSM output by removing tags
                if "<eng><asr>" in transcript:
                    transcript = transcript.replace("<eng><asr>", "")
                
            # Check if it's an ESPnet model
            elif "ESPnet" in asr_model_name:
                nbests = model(audio_data)
                transcript = nbests[0][0]  # Get the best hypothesis
            else:
                # Use transformers pipeline
                result = model({"array": audio_data, "sampling_rate": sr})
                transcript = result["text"]
        
    except Exception as e:
        print(f"Error in ASR transcription: {e}")
        transcript = "Error transcribing audio. Please try again."
    
    latency_ASR = time.time() - start_time
    return transcript

def generate_response(transcript, llm_model_name, system_prompt):
    """Generate response using selected LLM model"""
    global latency_LLM
    
    start_time = time.time()
    print(f"Starting LLM response generation with model: {llm_model_name}")
    
    try:
        model_info = load_llm_model(llm_model_name)
        
        if model_info is None:
            print(f"LLM model {llm_model_name} is unavailable")
            response = "LLM model unavailable. Please try a different model."
        else:
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Create input prompt
            prompt = f"{system_prompt}\nUser: {transcript}\nAssistant:"
            
            # Generate text
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            
            # Use the proper device
            if device == "cuda":
                input_ids = input_ids.to(torch_device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
    
    except Exception as e:
        print(f"Error in LLM response generation: {e}")
        response = "I'm sorry, I encountered an error generating a response. Please try again."
    
    latency_LLM = time.time() - start_time
    print(f"LLM response generation completed in {latency_LLM:.2f} seconds")
    return response

def clear_audio_input(clear=True):
    """Clear the audio input if the clear parameter is True"""
    if clear:
        print("Clearing audio input")
        return None
    else:
        return None

def process_speech_and_clear(audio_input, asr_option, llm_option, system_prompt, tts_option=None, clear=True):
    """Process speech and clear audio input afterwards if clear is True"""
    global latency_LLM

    print("Starting speech processing pipeline")
    
    # Check if audio input is available
    if audio_input is None:
        print("No audio input provided")
        return None, "", "", None
    
    try:
        # Get user input (transcribe speech)
        transcript = get_user_input(audio_input, asr_option)

        
        # Add user message to conversation history
        if transcript:
            controller.add_user_input(transcript)
            conversation_history.append({"role": "user", "content": transcript})
        
        # Generate response from agent
        start_time = time.time()
        action, observation = controller.get_next_chat_action()
        latency_LLM = time.time() - start_time
        response = action.payload
        # if isinstance(action, ChatAction):
        #     response = action.payload
        # else:
        #     response = "I have created a calendar event for you. Aura, signing off!"
        
        # Add assistant message to conversation history
        if response :
            conversation_history.append({"role": "assistant", "content": response})
        
        # Convert response to speech
        output_audio = agent_output(response, tts_option)
        
        # Clear audio input if needed
        if clear:
            input_audio = None
            print("🎤 Ready to record your response!")
        else:
            input_audio = audio_input
        
        print("Speech processing pipeline completed successfully")
        return input_audio, transcript, response, output_audio
    except Exception as e:
        print(f"Error in process_speech_and_clear: {e}")
        return None, f"Error processing audio: {str(e)}", "I couldn't process your speech properly. Please try again.", None

def display_latency():
    """Display latency information"""
    return f"""
    ASR Latency: {latency_ASR:.2f} seconds
    LLM Latency: {latency_LLM:.2f} seconds
    TTS Latency: {latency_TTS:.2f} seconds
    Total Latency: {latency_ASR + latency_LLM + latency_TTS:.2f} seconds
    """

def reset_conversation():
    """Reset the conversation history"""
    global conversation_history
    print("Resetting conversation history")
    conversation_history = []
    controller.reset()
    return None, "", "", None, ""

def display_conversation():
    """Display the conversation history with a simplified monochrome design and emojis"""
    if not conversation_history:
        return """
        <div style="background-color: #f0f5fa; border-radius: 8px; padding: 16px; font-family: 'Segoe UI', Arial, sans-serif;">
            <p style="color: #6e7c91; text-align: center; font-size: 14px;">No conversation yet. Start by recording your message.</p>
        </div>
        """
    output = """
    <div style="background-color: #f0f5fa; border-radius: 8px; padding: 16px; max-height: 500px; overflow-y: auto; font-family: 'Segoe UI', Arial, sans-serif;">
    """
    for item in conversation_history:
        role = item["role"]
        content = item["content"]
        
        if role == "user":
            output += f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; align-items: flex-start;">
                    <div style="margin-right: 8px; font-size: 16px;">👤</div>
                    <div style="background-color: #e1ebf7; border-radius: 8px; padding: 10px; max-width: 90%;">
                        <p style="margin: 0; color: #2c3e50; line-height: 1.4;">{content}</p>
                    </div>
                </div>
            </div>
            """
        else:
            output += f"""
            <div style="margin-bottom: 12px; display: flex; justify-content: flex-end;">
                <div style="display: flex; align-items: flex-start; flex-direction: row-reverse;">
                    <div style="margin-left: 8px; font-size: 16px;">🤖</div>
                    <div style="background-color: #d8e6f3; border-radius: 8px; padding: 10px; max-width: 90%;">
                        <p style="margin: 0; color: #2c3e50; line-height: 1.4;">{content}</p>
                    </div>
                </div>
            </div>
            """
    
    output += """
    </div>
    <style>
        /* Simple scrollbar styling */
        div::-webkit-scrollbar {
            width: 6px;
        }
        
        div::-webkit-scrollbar-track {
            background: #eef2f7;
            border-radius: 6px;
        }
        
        div::-webkit-scrollbar-thumb {
            background: #b8c6db;
            border-radius: 6px;
        }
        
        div::-webkit-scrollbar-thumb:hover {
            background: #99a9bf;
        }
    </style>
    """
    
    return output

def display_record_message(show=False):
    """Display a message prompting the user to record their response"""
    if show:
        return """
        <div style="background-color: #e7f7ed; border-radius: 8px; padding: 12px; margin-bottom: 15px; 
                font-family: 'Segoe UI', Arial, sans-serif; border-left: 4px solid #10b981;
                animation: fadeIn 0.5s ease-in-out; display: flex; align-items: center;">
            <span style="font-size: 20px; margin-right: 10px;">🎤</span>
            <p style="margin: 0; color: #0f766e; font-weight: 500;">Ready to record your response!</p>
            <span style="font-size: 20px; margin-left: 10px;"></span>
        </div>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """
    else:
        return ""

def create_demo():
    with gr.Blocks(title="Speech Conversation System", css="""
        .conversation-container {border: 1px solid #ddd; border-radius: 10px; overflow: hidden;}
        .conversation-container h3 {background-color: #f5f5f5; padding: 12px; margin: 0; border-bottom: 1px solid #ddd;}
        /* Hide submit button */
        #submit-btn {display: none;}
    """) as demo:
        gr.Markdown(
            """
            # Speech-based Task Oriented Dialogue System with Tool Use Integration
            
            This demo showcases a speech-to-speech task-oriented dialogue agent that results in tool use
            
            **Record your speech, and it will be automatically processed as soon as you stop recording.**
            """
        )
        
        record_message = gr.HTML()
        
        with gr.Row():
            with gr.Column(scale=2):
                record_message = gr.HTML()
                
                audio_input = gr.Audio(
                    label="Record your speech",
                    sources=["microphone"],
                    type="numpy",
                    streaming=False,
                    autoplay=False
                )
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful AI assistant."
                )
                
                with gr.Row():
                    asr_dropdown = gr.Dropdown(
                        choices=list(ASR_OPTIONS.keys()),
                        value=list(ASR_OPTIONS.keys())[0],
                        label="Select ASR Model"
                    )
                    
                    llm_dropdown = gr.Dropdown(
                        choices=list(LLM_OPTIONS.keys()),
                        value=list(LLM_OPTIONS.keys())[0],
                        label="Select Agent"
                    )
                
                tts_choices = list(ESPNET_TTS_OPTIONS.keys()) 
                tts_dropdown = gr.Dropdown(
                    choices=tts_choices,
                    value=tts_choices[0],
                    label="Select TTS Model"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Process Speech", variant="primary", elem_id="submit-btn")
                    reset_btn = gr.Button("Reset Conversation")
                
                user_transcript = gr.Textbox(label="Your Speech (ASR Output)")
                system_response = gr.Textbox(label="AI Response (LLM Output)")
                audio_output = gr.Audio(label="AI Voice Response", autoplay=True)
                latency_info = gr.Textbox(label="Performance Metrics")
            
            with gr.Column(scale=1, elem_classes="conversation-container"):
                conversation_display = gr.HTML(label="Conversation History")
        
        def show_record_message():
            return display_record_message(True)
            
        def hide_record_message():
            return display_record_message(False)
        
        t = True
        
        # Event handlers - Process speech automatically when audio is recorded
        audio_input.stop_recording(
            fn=process_speech_and_clear,
            inputs=[audio_input, asr_dropdown, llm_dropdown, system_prompt, tts_dropdown],
            outputs=[audio_input, user_transcript, system_response, audio_output]  
        ).then(
            display_latency,
            inputs=[],
            outputs=[latency_info]
        ).then(
            display_conversation,  # Display updated conversation without adding new messages
            inputs=[],
            outputs=[conversation_display]
        ).then(
            fn=lambda: show_record_message() if t else "",
            inputs=[],
            outputs=[record_message]
        )
        
        # Hide the record message when a new recording starts
        audio_input.start_recording(
            fn=lambda: hide_record_message() if t else "",
            inputs=[],
            outputs=[record_message]
        )
        
        
        reset_btn.click(
            reset_conversation,
            inputs=[],
            outputs=[audio_input, user_transcript, system_response, audio_output, latency_info]
        ).then(
            display_conversation,
            inputs=[],
            outputs=[conversation_display]
        ).then(
            fn=lambda: show_record_message() if t else "" , 
            inputs=[],
            outputs=[record_message]
        )
    
    return demo

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Speech Conversation System")
    print(f"Running in {'GPU' if device == 'cuda' else 'CPU'} mode")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 50)
    
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=True
    )