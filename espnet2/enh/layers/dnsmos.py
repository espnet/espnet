import json
import math

import librosa
import numpy as np
import requests
import torch
import torchaudio

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
# URL for the web service
SCORING_URI_DNSMOS = "https://dnsmos.azurewebsites.net/score"
SCORING_URI_DNSMOS_P835 = "https://dnsmos.azurewebsites.net/v1/dnsmosp835/score"


def poly1d(coefficients, use_numpy=False):
    if use_numpy:
        return np.poly1d(coefficients)
    coefficients = tuple(reversed(coefficients))

    def func(p):
        return sum(coef * p**i for i, coef in enumerate(coefficients))

    return func


class DNSMOS_web:
    # ported from
    # https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos.py
    def __init__(self, auth_key):
        self.auth_key = auth_key

    def __call__(self, aud, input_fs, fname="", method="p808"):
        if input_fs != SAMPLING_RATE:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE)
        else:
            audio = aud

        # Set the content type
        headers = {"Content-Type": "application/json"}
        # If authentication is enabled, set the authorization header
        headers["Authorization"] = f"Basic {self.auth_key}"
        fname = fname + ".wav" if fname else "audio.wav"
        data = {"data": audio.tolist(), "filename": fname}
        input_data = json.dumps(data)
        # Make the request and display the response
        if method == "p808":
            u = SCORING_URI_DNSMOS
        else:
            u = SCORING_URI_DNSMOS_P835
        resp = requests.post(u, data=input_data, headers=headers)
        score_dict = resp.json()
        return score_dict


class DNSMOS_local:
    # ported from
    # https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
    def __init__(
        self, primary_model_path, p808_model_path, use_gpu=False, convert_to_torch=False
    ):
        self.convert_to_torch = convert_to_torch
        self.use_gpu = use_gpu
        if convert_to_torch:
            try:
                from onnx2torch import convert
            except ModuleNotFoundError:
                raise RuntimeError("Please install onnx2torch manually and retry!")

            if primary_model_path is not None:
                self.primary_model = convert(primary_model_path).eval()
                self.p808_model = convert(p808_model_path).eval()
            self.spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=321, hop_length=160, pad_mode="constant"
            )

            self.to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80.0)
            if use_gpu:
                if primary_model_path is not None:
                    self.primary_model = self.primary_model.cuda()
                    self.p808_model = self.p808_model.cuda()
                self.spectrogram = self.spectrogram.cuda()
        else:
            try:
                import onnxruntime as ort
            except ModuleNotFoundError:
                raise RuntimeError("Please install onnxruntime manually and retry!")

            prvd = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
            if primary_model_path is not None:
                self.onnx_sess = ort.InferenceSession(
                    primary_model_path, providers=[prvd]
                )
                self.p808_onnx_sess = ort.InferenceSession(
                    p808_model_path, providers=[prvd]
                )

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        if self.convert_to_torch:
            specgram = self.spectrogram(audio)
            fb = torch.as_tensor(
                librosa.filters.mel(sr=sr, n_fft=frame_size + 1, n_mels=n_mels).T,
                dtype=audio.dtype,
                device=audio.device,
            )
            mel_spec = torch.matmul(specgram.transpose(-1, -2), fb).transpose(-1, -2)
            if to_db:
                self.to_db.db_multiplier = math.log10(
                    max(self.to_db.amin, torch.max(mel_spec))
                )
                mel_spec = (self.to_db(mel_spec) + 40) / 40
        else:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=frame_size + 1,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            if to_db:
                mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        flag = not self.convert_to_torch
        if is_personalized_MOS:
            p_ovr = poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046], flag)
            p_sig = poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726], flag)
            p_bak = poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132], flag)
        else:
            p_ovr = poly1d([-0.06766283, 1.11546468, 0.04602535], flag)
            p_sig = poly1d([-0.08397278, 1.22083953, 0.0052439], flag)
            p_bak = poly1d([-0.13166888, 1.60915514, -0.39604546], flag)

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs, is_personalized_MOS=False):
        if self.convert_to_torch:
            device = "cuda" if self.use_gpu else "cpu"
            if isinstance(aud, torch.Tensor):
                aud = aud.to(device=device)
            else:
                aud = torch.as_tensor(aud, dtype=torch.float32, device=device)
        else:
            aud = aud.cpu().detach().numpy() if isinstance(aud, torch.Tensor) else aud
        if input_fs != SAMPLING_RATE:
            if self.convert_to_torch:
                audio = torch.as_tensor(
                    librosa.resample(
                        aud.detach().cpu().numpy(),
                        orig_sr=input_fs,
                        target_sr=SAMPLING_RATE,
                    ),
                    dtype=aud.dtype,
                    device=aud.device,
                )
            else:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE)
        else:
            audio = aud
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            if self.convert_to_torch:
                audio = torch.cat((audio, audio))
            else:
                audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            if self.convert_to_torch:
                input_features = audio_seg.float()[None, :]
                p808_input_features = self.audio_melspec(
                    audio=audio_seg[:-160]
                ).float()[None, :, :]
                p808_mos = self.p808_model(p808_input_features)
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.primary_model(
                    input_features
                )[0]
            else:
                input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
                p808_input_features = np.array(
                    self.audio_melspec(audio=audio_seg[:-160])
                ).astype("float32")[np.newaxis, :, :]
                p808_mos = self.p808_onnx_sess.run(
                    None, {"input_1": p808_input_features}
                )[0][0][0]
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(
                    None, {"input_1": input_features}
                )[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        to_array = torch.stack if self.convert_to_torch else np.array
        return {
            "OVRL_raw": to_array(predicted_mos_ovr_seg_raw).mean(),
            "SIG_raw": to_array(predicted_mos_sig_seg_raw).mean(),
            "BAK_raw": to_array(predicted_mos_bak_seg_raw).mean(),
            "OVRL": to_array(predicted_mos_ovr_seg).mean(),
            "SIG": to_array(predicted_mos_sig_seg).mean(),
            "BAK": to_array(predicted_mos_bak_seg).mean(),
            "P808_MOS": to_array(predicted_p808_mos).mean(),
        }
