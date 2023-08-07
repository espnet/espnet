import json

import librosa
import numpy as np
import requests

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
# URL for the web service
SCORING_URI_DNSMOS = "https://dnsmos.azurewebsites.net/score"
SCORING_URI_DNSMOS_P835 = "https://dnsmos.azurewebsites.net/v1/dnsmosp835/score"


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
    def __init__(self, primary_model_path, p808_model_path, use_gpu=False):
        import onnxruntime as ort

        prvd = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=[prvd])
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=[prvd])

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs, is_personalized_MOS=False):
        if input_fs != SAMPLING_RATE:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE)
        else:
            audio = aud
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
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

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
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

        return {
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
            "P808_MOS": np.mean(predicted_p808_mos),
        }
