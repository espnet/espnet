import librosa
import torch

wave, sr = librosa.load(
    "/ocean/projects/cis210027p/yzhao16/speechlm/data_format2/espnet_fork/egs2/opencpop/speechlm1/downloads/raw_data/wavs/2001000001.wav",
    sr=None,
    mono=True,
)
predictor = torch.hub.load(
    "South-Twilight/SingMOS:v0.2.0", "singsing-ssl-mos", trust_repo=True
)
wave = torch.from_numpy(wave)
length = torch.tensor([wave.shape[1]])
# wave: [B, T], length: [B]
score = predictor(wave, length)
# tensor([3.7730])
