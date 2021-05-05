#!/usr/bin/env python3

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("localhost", 9988))
s.listen(1)

fs, lang = 22050, "English"
tag = "kan-bayashi/ljspeech_conformer_fastspeech2"
vocoder_tag = "ljspeech_full_band_melgan.v2"

import time
import os
import torch
import soundfile as sf
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
d = ModelDownloader()
text2speech = Text2Speech(
    **d.download_and_unpack(tag),
    device="cuda",
    speed_control_alpha=1.0,
)
text2speech.spc2wav = None  # Disable griffin-lim
vocoder = load_model(download_pretrained_model(vocoder_tag)).to("cuda").eval()
vocoder.remove_weight_norm()

while True:
    conn, addr = s.accept()
    data = conn.recv(1024)
    encoding = 'utf-8'
    data = str(data, encoding)
    conn.close()
    # synthesis
    with torch.no_grad():
        start = time.time()
        wav, c, *_ = text2speech(data)
        wav = vocoder.inference(c)
    rtf = (time.time() - start) / (len(wav) / fs)
    print(f"RTF = {rtf:5f}")
    
    # let us listen to generated samples
    
    sf.write("out2.wav", wav.view(-1).cpu().numpy(), fs, "PCM_16")
    file = "out2.wav"
    os.system("aplay " + file)
    
