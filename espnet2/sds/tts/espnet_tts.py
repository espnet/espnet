from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from espnet2.sds.tts.abs_tts import AbsTTS
from espnet_model_zoo.downloader import ModelDownloader
import os
import numpy as np
import torch
from typeguard import typechecked
import glob

class ESPnetTTSModel(AbsTTS):
    """ESPnet TTS."""

    @typechecked
    def __init__(
        self,
        tag = 'kan-bayashi/ljspeech_vits',
        device="cuda",
    ): 
        super().__init__()
        vocoder_tag = "none"
        self.d = ModelDownloader()
        self.sids = None
        self.spembs = None
        self.text2speech = Text2Speech.from_pretrained(
            model_tag=tag,
            vocoder_tag=str_or_none(vocoder_tag),
            device=device,
            # Only for Tacotron 2 & Transformer
            threshold=0.5,
            # Only for Tacotron 2
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3,
            # Only for FastSpeech & FastSpeech2 & VITS
            speed_control_alpha=1.0,
            # Only for VITS
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )
        model_dir = os.path.dirname(self.d.download_and_unpack(tag)["train_config"])
        if self.text2speech.use_sids:
            spk2sid = glob.glob(f"{model_dir}/../../dump/**/spk2sid", recursive=True)[0]
            with open(spk2sid) as f:
                lines = [line.strip() for line in f.readlines()]
            sid2spk = {int(line.split()[1]): line.split()[0] for line in lines}
            
            # randomly select speaker
            self.sids = np.array(np.random.randint(1, len(sid2spk)))
        if self.text2speech.use_spembs:
            import kaldiio
            xvector_ark = [p for p in glob.glob(f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
            xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
            spks = list(xvectors.keys())

            # randomly select speaker
            random_spk_idx = np.random.randint(0, len(spks))
            spk = spks[random_spk_idx]
            self.spembs = xvectors[spk]
    
    def warmup(self):
        with torch.no_grad():
            wav = self.text2speech("Sid", sids=self.sids, spembs=self.spembs)["wav"]
    
    def forward(self,transcript):
        with torch.no_grad():
            audio_chunk = self.text2speech(transcript, sids=self.sids, spembs=self.spembs)["wav"].view(-1).cpu().numpy()
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            return (self.text2speech.fs, audio_chunk)