"""External VAD models as frame-wise speech probabilities, for local/baselines/run.py.

Every model maps a 16 kHz mono float32 waveform to one speech probability per frame;
`shift` is the time between frames and `offset` the centre of frame 0 (seconds), as
local/v2/score_vad.py score --shift --offset takes them. `thresholds` and `medians`
(centred, in frames of the model) are the grid tuned on the tune set
(local/baselines/eval.sh). Versions and model revisions are pinned in
local/baselines/requirements.txt and below.

  silero     Silero VAD (silero-vad package, its bundled model), 32 ms frames
  webrtc0    WebRTC VAD, aggressiveness 0, 0/1 decisions on 30 ms frames
  marblenet  NVIDIA Frame_VAD_Multilingual_MarbleNet_v2.0 (NeMo), 20 ms frames
"""

import os

import numpy as np

FS = 16000
GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MARBLENET_REPO = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"
MARBLENET_FILE = "frame_vad_multilingual_marblenet_v2.0.nemo"
MARBLENET_REVISION = "9d25eb877d9edab5101ccc843edd04787a6c72c4"


class Silero:
    shift, offset = 512 / FS, 256 / FS
    thresholds, medians = GRID, (1, 3)
    gpu = False

    def __init__(self, model_dir=None):
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(int(os.environ.get("THREADS", "1")))
        self.model = load_silero_vad()  # the TorchScript model in the package

    def __call__(self, wav):
        import torch

        prob = self.model.audio_forward(torch.from_numpy(wav), sr=FS)[0].numpy()
        self.model.reset_states()
        return prob


class WebRTC:
    shift, offset = 480 / FS, 240 / FS
    thresholds, medians = (0.5,), (1, 3)
    gpu = False

    def __init__(self, model_dir=None, mode=0):
        import webrtcvad

        self.vad = webrtcvad.Vad(mode)

    def __call__(self, wav):
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
        n = len(pcm) // 480
        return np.array(
            [
                self.vad.is_speech(pcm[i * 480 : (i + 1) * 480].tobytes(), FS)
                for i in range(n)
            ],
            dtype=np.float32,
        )


class MarbleNet:
    shift, offset = 0.02, 0.01
    thresholds, medians = GRID, (1, 5)
    gpu = True
    chunk = 600 * FS  # 10-min pieces keep long recordings within GPU memory

    def __init__(self, model_dir=None):
        import torch
        from nemo.collections.asr.models import EncDecFrameClassificationModel

        if model_dir:  # an offline copy of the .nemo file
            path = os.path.join(model_dir, MARBLENET_FILE)
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                MARBLENET_REPO, MARBLENET_FILE, revision=MARBLENET_REVISION
            )
        # strict=False: the released .nemo has no "loss.weight" (the training loss's
        # class weights), which NeMo 3.0 expects; inference does not use it
        self.model = EncDecFrameClassificationModel.restore_from(path, strict=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.eval().to(self.device)

    def __call__(self, wav):
        import torch

        probs = []
        with torch.no_grad():
            for s in range(0, len(wav), self.chunk):
                x = torch.from_numpy(wav[s : s + self.chunk]).to(self.device)[None]
                n = torch.tensor([x.shape[1]], device=self.device)
                logits = self.model(input_signal=x, input_signal_length=n)
                probs.append(
                    torch.softmax(logits[0], dim=-1)[:, 1].float().cpu().numpy()
                )
        return np.concatenate(probs)


MODELS = {"silero": Silero, "webrtc0": WebRTC, "marblenet": MarbleNet}
