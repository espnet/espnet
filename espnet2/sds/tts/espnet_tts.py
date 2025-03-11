import glob
import os
from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.bin.tts_inference import Text2Speech
from espnet2.sds.tts.abs_tts import AbsTTS
from espnet2.utils.types import str_or_none


class ESPnetTTSModel(AbsTTS):
    """ESPnet TTS."""

    @typechecked
    def __init__(
        self,
        tag: str = "kan-bayashi/ljspeech_vits",
        device: str = "cuda",
    ):
        """A class to initialize and manage a ESPnet's

        pre-trained text-to-speech (TTS) model.

        This class:
        1. Downloads and sets up a pre-trained TTS model using the ESPnet Model Zoo.
        2. Supports various TTS configurations, including multi speaker TTS
        using speaker embeddings and speaker IDs.

        Args:
            tag (str, optional):
                The model tag for the pre-trained TTS model.
                Defaults to "kan-bayashi/ljspeech_vits".
            device (str, optional):
                The computation device for running inference.
                Defaults to "cuda".

        Raises:
            ImportError:
                If the required `espnet_model_zoo` library is not installed.
        """
        try:
            from espnet_model_zoo.downloader import ModelDownloader
        except Exception as e:
            print("Error: espnet_model_zoo is not properly installed.")
            raise e
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

            xvector_ark = [
                p
                for p in glob.glob(
                    f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True
                )
                if "tr" in p
            ][0]
            xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
            spks = list(xvectors.keys())

            # randomly select speaker
            random_spk_idx = np.random.randint(0, len(spks))
            spk = spks[random_spk_idx]
            self.spembs = xvectors[spk]

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
        with torch.no_grad():
            _ = self.text2speech("Sid", sids=self.sids, spembs=self.spembs)["wav"]

    def forward(self, transcript: str) -> Tuple[int, np.ndarray]:
        """Converts a text transcript into an audio waveform

        using a pre-trained ESPnet-TTS model.

        Args:
            transcript (str):
                The input text to be converted into speech.

        Returns:
            Tuple[int, np.ndarray]:
                A tuple containing:
                - The sample rate of the audio (int).
                - The generated audio waveform as a
                NumPy array of type `int16`.
        """
        with torch.no_grad():
            audio_chunk = (
                self.text2speech(transcript, sids=self.sids, spembs=self.spembs)["wav"]
                .view(-1)
                .cpu()
                .numpy()
            )
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            return (self.text2speech.fs, audio_chunk)
