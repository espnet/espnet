import json
import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import scipy.signal
import soundfile
from typeguard import check_argument_types, check_return_type

from espnet2.layers.augmentation import DataAugmentation
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer


class AbsPreprocessor(ABC):
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


def any_allzero(signal):
    if isinstance(signal, (list, tuple)):
        return any([np.allclose(s, 0.0) for s in signal])
    return np.allclose(signal, 0.0)


class CommonPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        use_lang_prompt: bool = False,
        use_nlp_prompt: bool = False,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        fs: int = 0,
        nonsplit_symbol: Iterable[str] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        # only use for whisper
        whisper_language: str = None,
        whisper_task: str = None,
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob
        self.short_noise_thres = short_noise_thres
        self.aux_task_names = aux_task_names
        self.use_lang_prompt = use_lang_prompt
        self.use_nlp_prompt = use_nlp_prompt

        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
                nonsplit_symbol=nonsplit_symbol,
                whisper_language=whisper_language,
                whisper_task=whisper_task,
            )
            if token_type == "hugging_face":
                self.token_id_converter = HuggingFaceTokenIDConverter(
                    model_name_or_path=bpemodel
                )
            elif bpemodel not in ["whisper_en", "whisper_multilingual"]:
                self.token_id_converter = TokenIDConverter(
                    token_list=token_list,
                    unk_symbol=unk_symbol,
                )
            else:
                self.token_id_converter = OpenAIWhisperTokenIDConverter(
                    model_type=bpemodel,
                    added_tokens_txt=non_linguistic_symbols,
                    language=whisper_language or "en",
                    task=whisper_task or "transcribe",
                )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            rir_scp = [rir_scp] if not isinstance(rir_scp, (list, tuple)) else rir_scp
            for scp in rir_scp:
                with open(scp, "r", encoding="utf-8") as f:
                    for line in f:
                        sps = line.strip().split(None, 1)
                        if len(sps) == 1:
                            self.rirs.append(sps[0])
                        else:
                            self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            noise_scp = (
                [noise_scp] if not isinstance(noise_scp, (list, tuple)) else noise_scp
            )
            for scp in noise_scp:
                with open(scp, "r", encoding="utf-8") as f:
                    for line in f:
                        sps = line.strip().split(None, 1)
                        if len(sps) == 1:
                            self.noises.append(sps[0])
                        else:
                            self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low = self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

        # Check DataAugmentation docstring for more information of `data_aug_effects`
        self.fs = fs
        if data_aug_effects is not None:
            assert self.fs > 0, self.fs
            self.data_aug = DataAugmentation(data_aug_effects, apply_n=data_aug_num)
        else:
            self.data_aug = None
        self.data_aug_prob = data_aug_prob

    def _convolve_rir(self, speech, power, rirs, tgt_fs=None, single_channel=False):
        rir_path = np.random.choice(rirs)
        rir = None
        if rir_path is not None:
            rir, fs = soundfile.read(rir_path, dtype=np.float64, always_2d=True)

            if single_channel:
                num_ch = rir.shape[1]
                chs = [np.random.randint(num_ch)]
                rir = rir[:, chs]
            # rir: (Nmic, Time)
            rir = rir.T
            if tgt_fs and fs != tgt_fs:
                logging.warning(
                    f"Resampling RIR to match the sampling rate ({fs} -> {tgt_fs} Hz)"
                )
                rir = librosa.resample(
                    rir, orig_sr=fs, target_sr=tgt_fs, res_type="kaiser_fast"
                )

            # speech: (Nmic, Time)
            speech = speech[:1]
            # Note that this operation doesn't change the signal length
            speech = scipy.signal.convolve(speech, rir, mode="full")[
                :, : speech.shape[1]
            ]
            # Reverse mean power to the original power
            power2 = (speech[detect_non_silence(speech)] ** 2).mean()
            speech = np.sqrt(power / max(power2, 1e-10)) * speech
        return speech, rir

    def _add_noise(
        self,
        speech,
        power,
        noises,
        noise_db_low,
        noise_db_high,
        tgt_fs=None,
        single_channel=False,
    ):
        nsamples = speech.shape[1]
        noise_path = np.random.choice(noises)
        noise = None
        if noise_path is not None:
            noise_db = np.random.uniform(noise_db_low, noise_db_high)
            with soundfile.SoundFile(noise_path) as f:
                fs = f.samplerate
                if tgt_fs and fs != tgt_fs:
                    nsamples_ = int(nsamples / tgt_fs * fs) + 1
                else:
                    nsamples_ = nsamples
                if f.frames == nsamples_:
                    noise = f.read(dtype=np.float64, always_2d=True)
                elif f.frames < nsamples_:
                    if f.frames / nsamples_ < self.short_noise_thres:
                        logging.warning(
                            f"Noise ({f.frames}) is much shorter than "
                            f"speech ({nsamples_}) in dynamic mixing"
                        )
                    offset = np.random.randint(0, nsamples_ - f.frames)
                    # noise: (Time, Nmic)
                    noise = f.read(dtype=np.float64, always_2d=True)
                    # Repeat noise
                    noise = np.pad(
                        noise,
                        [(offset, nsamples_ - f.frames - offset), (0, 0)],
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples_)
                    f.seek(offset)
                    # noise: (Time, Nmic)
                    noise = f.read(nsamples_, dtype=np.float64, always_2d=True)
                    if len(noise) != nsamples_:
                        raise RuntimeError(f"Something wrong: {noise_path}")
            if single_channel:
                num_ch = noise.shape[1]
                chs = [np.random.randint(num_ch)]
                noise = noise[:, chs]
            # noise: (Nmic, Time)
            noise = noise.T
            if tgt_fs and fs != tgt_fs:
                logging.warning(
                    f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"
                )
                noise = librosa.resample(
                    noise, orig_sr=fs, target_sr=tgt_fs, res_type="kaiser_fast"
                )
                if noise.shape[1] < nsamples:
                    noise = np.pad(
                        noise, [(0, 0), (0, nsamples - noise.shape[1])], mode="wrap"
                    )
                else:
                    noise = noise[:, :nsamples]

            noise_power = (noise**2).mean()
            scale = (
                10 ** (-noise_db / 20)
                * np.sqrt(power)
                / np.sqrt(max(noise_power, 1e-10))
            )
            speech = speech + scale * noise
        return speech, noise

    def _speech_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        assert check_argument_types()
        if self.speech_name in data:
            if self.train and (self.rirs is not None or self.noises is not None):
                speech = data[self.speech_name]

                # speech: (Nmic, Time)
                if speech.ndim == 1:
                    speech = speech[None, :]
                else:
                    speech = speech.T
                # Calc power on non silence region
                power = (speech[detect_non_silence(speech)] ** 2).mean()

                # 1. Convolve RIR
                if self.rirs is not None and self.rir_apply_prob >= np.random.random():
                    speech, _ = self._convolve_rir(speech, power, self.rirs)

                # 2. Add Noise
                if (
                    self.noises is not None
                    and self.noise_apply_prob >= np.random.random()
                ):
                    speech, _ = self._add_noise(
                        speech,
                        power,
                        self.noises,
                        self.noise_db_low,
                        self.noise_db_high,
                    )

                speech = speech.T
                ma = np.max(np.abs(speech))
                if ma > 1.0:
                    speech /= ma
                data[self.speech_name] = speech

            if self.train and self.data_aug:
                if self.data_aug_prob > 0 and self.data_aug_prob >= np.random.random():
                    data[self.speech_name] = self.data_aug(
                        data[self.speech_name], self.fs
                    )

            if self.speech_volume_normalize is not None:
                speech = data[self.speech_name]
                ma = np.max(np.abs(speech))
                data[self.speech_name] = speech * self.speech_volume_normalize / ma
        assert check_return_type(data)
        return data

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            if isinstance(text, np.ndarray):
                return data
            text = self.text_cleaner(text)
            tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            if len(text_ints) > 500:
                logging.warning(
                    "The length of the text output exceeds 500, "
                    "which may cause OOM on the GPU."
                    "Please ensure that the data processing is correct and verify it."
                )
            if "prompt" in data:
                actual_token = (
                    self.token_id_converter.tokenizer.tokenizer.convert_ids_to_tokens(
                        text_ints
                    )
                )
                if self.use_lang_prompt:
                    if data["prompt"] == "<|nospeech|>":
                        actual_token = [data["prompt"]]
                    else:
                        actual_token = data["prompt"].split() + actual_token[2:]
                elif self.use_nlp_prompt:
                    prompt_tokens = self.tokenizer.text2tokens(data["prompt"])
                    actual_token = [actual_token[0]] + prompt_tokens + actual_token[2:]
                else:
                    if len(data["prompt"].split()) > 1:
                        actual_token = (
                            [actual_token[0]]
                            + data["prompt"].split()
                            + actual_token[2:]
                        )
                    else:
                        actual_token[1] = data["prompt"]
                text_ints = (
                    self.token_id_converter.tokenizer.tokenizer.convert_tokens_to_ids(
                        actual_token
                    )
                )
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
            if "prompt" in data:
                whisper_tokenizer = self.token_id_converter.tokenizer.tokenizer
                if len(data["prompt"].split()) > 1:
                    data["prompt"] = np.array(
                        whisper_tokenizer.convert_tokens_to_ids(data["prompt"].split()),
                        dtype=np.int64,
                    )
                else:
                    data["prompt"] = np.array(
                        [whisper_tokenizer.convert_tokens_to_ids(data["prompt"])],
                        dtype=np.int64,
                    )
        if self.aux_task_names is not None and self.tokenizer is not None:
            for name in self.aux_task_names:
                if name in data:
                    text = data[name]
                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.text2tokens(text)
                    text_ints = self.token_id_converter.tokens2ids(tokens)
                    data[name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._text_process(data)
        return data


class SLUPreprocessor(CommonPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        transcript_token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        fs: int = 0,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
    ):
        super().__init__(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            text_name=text_name,
            fs=fs,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
        )
        if transcript_token_list is not None:
            print("using transcript")
            self.transcript_tokenizer = build_tokenizer(
                token_type="word",
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.transcript_token_id_converter = TokenIDConverter(
                token_list=transcript_token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.transcript_tokenizer = None
            self.transcript_token_id_converter = None

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            text = self.text_cleaner(text)
            tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
        if "transcript" in data and self.tokenizer is not None:
            text = data["transcript"]
            text = self.text_cleaner(text)
            tokens = self.transcript_tokenizer.text2tokens(text)
            text_ints = self.transcript_token_id_converter.tokens2ids(tokens)
            data["transcript"] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data


class CommonPreprocessor_multi(CommonPreprocessor):
    def __init__(
        self,
        train: bool,
        use_lang_prompt: bool = False,
        use_nlp_prompt: bool = False,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: List[str] = ["text"],
        fs: int = 0,
        speaker_change_symbol: Iterable[str] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        # only use for whisper
        whisper_language: str = None,
        whisper_task: str = None,
    ):
        super().__init__(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            aux_task_names=aux_task_names,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            fs=fs,
            nonsplit_symbol=speaker_change_symbol,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
            whisper_language=whisper_language,
            whisper_task=whisper_task,
        )
        if isinstance(text_name, str):
            self.text_name = [text_name]
        else:
            self.text_name = text_name

        self.speaker_change_symbol = speaker_change_symbol
        if speaker_change_symbol is not None:
            assert (
                len(self.text_name) == 1
            ), "SOT model with speaker_change_symbol only support single text input."

            if bpemodel in ["whisper_en", "whisper_multilingual"]:
                assert (
                    len(speaker_change_symbol) == 1
                ), "Currently, Whisper SOT only supports one SC token"
                speaker_change_symbol = speaker_change_symbol[0]
                self.tokenizer = OpenAIWhisperTokenizer(
                    model_type=bpemodel,
                    language=whisper_language or "en",
                    task=whisper_task or "transcribe",
                    sot=True,
                    speaker_change_symbol=speaker_change_symbol,
                )
                self.token_id_converter = OpenAIWhisperTokenIDConverter(
                    model_type=bpemodel,
                    language=whisper_language or "en",
                    task=whisper_task or "transcribe",
                    sot=True,
                    speaker_change_symbol=speaker_change_symbol,
                )

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        for text_n in self.text_name:
            if text_n in data and self.tokenizer is not None:
                text = data[text_n]
                text = self.text_cleaner(text)
                tokens = self.tokenizer.text2tokens(text)
                text_ints = self.token_id_converter.tokens2ids(tokens)
                data[text_n] = np.array(text_ints, dtype=np.int64)
        if self.aux_task_names is not None and self.tokenizer is not None:
            for name in self.aux_task_names:
                if name in data:
                    text = data[name]
                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.text2tokens(text)
                    text_ints = self.token_id_converter.tokens2ids(tokens)
                    data[name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._text_process(data)
        return data


class MutliTokenizerCommonPreprocessor(CommonPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: List[str] = [None],
        token_list: List[Union[Path, str, Iterable[str]]] = [None],
        bpemodel: List[Union[Path, str, Iterable[str]]] = [None],
        text_cleaner: Collection[str] = None,
        g2p_type: Union[List[str], str] = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: List[str] = ["text"],
        tokenizer_encode_conf: List[Dict] = [dict(), dict()],
        fs: int = 0,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        # only use for whisper
        whisper_language: List[str] = None,
        whisper_task: str = None,
    ):
        # TODO(jiatong): sync with Kamo and Jing on interface for preprocessor
        super().__init__(
            train=train,
            token_type=token_type[0],
            token_list=token_list[0],
            bpemodel=bpemodel[0],
            text_cleaner=text_cleaner,
            g2p_type=g2p_type[0]
            if type(g2p_type) is not str and g2p_type is not None
            else g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            speech_name=speech_name,
            text_name=text_name[0],
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            fs=fs,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
        )

        assert (
            len(token_type) == len(token_list) == len(bpemodel) == len(text_name)
        ), "token_type, token_list, bpemodel, or processing text_name mismatched"
        self.num_tokenizer = len(token_type)
        self.tokenizer = []
        self.token_id_converter = []

        if type(g2p_type) is str:
            # NOTE(jiatong): str will repeat for every tokenizer
            g2p_type = [g2p_type] * self.num_tokenizer

        for i in range(self.num_tokenizer):
            if token_type[i] is not None:
                if token_list[i] is None:
                    raise ValueError("token_list is required if token_type is not None")

                self.tokenizer.append(
                    build_tokenizer(
                        token_type=token_type[i],
                        bpemodel=bpemodel[i],
                        delimiter=delimiter,
                        space_symbol=space_symbol,
                        non_linguistic_symbols=non_linguistic_symbols,
                        g2p_type=g2p_type[i] if g2p_type is not None else g2p_type,
                        encode_kwargs=(
                            tokenizer_encode_conf[i]
                            if i < len(tokenizer_encode_conf)
                            else None
                        ),
                        whisper_language=whisper_language[i]
                        if "whisper" in token_type[i]
                        else None,
                        whisper_task=whisper_task,
                    )
                )

                if "whisper" not in token_type[i]:
                    self.token_id_converter.append(
                        TokenIDConverter(
                            token_list=token_list[i],
                            unk_symbol=unk_symbol,
                        )
                    )
                else:
                    self.token_id_converter.append(
                        OpenAIWhisperTokenIDConverter(
                            model_type=bpemodel[i],
                            language=whisper_language[i] or "en",
                            task=whisper_task or "translate",
                        )
                    )
            else:
                self.tokenizer.append(None)
                self.token_id_converter.append(None)

        self.text_cleaner = TextCleaner(text_cleaner)
        self.text_name = text_name  # override the text_name from CommonPreprocessor

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        for i in range(self.num_tokenizer):
            text_name = self.text_name[i]
            if text_name in data and self.tokenizer[i] is not None:
                text = data[text_name]
                text = self.text_cleaner(text)
                tokens = self.tokenizer[i].text2tokens(text)
                text_ints = self.token_id_converter[i].tokens2ids(tokens)
                data[text_name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data


class DynamicMixingPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        source_scp: str = None,
        ref_num: int = 2,
        dynamic_mixing_gain_db: float = 0.0,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        mixture_source_name: str = None,
        utt2spk: str = None,
        categories: Optional[List] = None,
    ):
        super().__init__(train)
        self.source_scp = source_scp
        self.ref_num = ref_num
        self.dynamic_mixing_gain_db = dynamic_mixing_gain_db
        self.speech_name = speech_name
        self.speech_ref_name_prefix = speech_ref_name_prefix
        # mixture_source_name: the key to select source utterances from dataloader
        if mixture_source_name is None:
            self.mixture_source_name = f"{speech_ref_name_prefix}1"
        else:
            self.mixture_source_name = mixture_source_name

        self.sources = {}
        assert (
            source_scp is not None
        ), f"Please pass `source_scp` to {type(self).__name__}"
        with open(source_scp, "r", encoding="utf-8") as f:
            for line in f:
                sps = line.strip().split(None, 1)
                assert len(sps) == 2
                self.sources[sps[0]] = sps[1]

        self.utt2spk = {}
        if utt2spk is None:
            # if utt2spk is not provided, create a dummy utt2spk with uid.
            for key in self.sources.keys():
                self.utt2spk[key] = key
        else:
            with open(utt2spk, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    assert len(sps) == 2
                    self.utt2spk[sps[0]] = sps[1]

            for key in self.sources.keys():
                assert key in self.utt2spk

        self.source_keys = list(self.sources.keys())

        # Map each category into a unique integer
        self.categories = {}
        if categories:
            count = 0
            for c in categories:
                if c not in self.categories:
                    self.categories[c] = count
                    count += 1

    def _pick_source_utterances_(self, uid):
        # return (ref_num - 1) uid of reference sources.

        source_keys = [uid]

        spk_ids = [self.utt2spk[uid]]

        retry_cnt = 0
        while len(source_keys) < self.ref_num:
            picked = random.choice(self.source_keys)
            spk_id = self.utt2spk[picked]

            # make one utterance or one speaker only appears once in mixing.
            if (picked not in source_keys) and (spk_id not in spk_ids):
                source_keys.append(picked)
            else:
                retry_cnt += 1
                if retry_cnt > 10:
                    source_keys.append(picked)
                    logging.warning(
                        "Can not find speech source from different speaker "
                        f"for {retry_cnt} times."
                        "There may be problems with training data. "
                        "Please check the utt2spk file."
                    )

        return source_keys[1:]

    def _read_source_(self, key, speech_length):
        source, _ = soundfile.read(
            self.sources[key],
            dtype=np.float32,
            always_2d=False,
        )

        if speech_length > source.shape[0]:
            pad = speech_length - source.shape[0]
            source = np.pad(source, (0, pad), "reflect")
        else:
            source = source[0:speech_length]

        assert speech_length == source.shape[0]

        return source

    def _mix_speech_(self, uid, data):
        # pick sources
        source_keys = self._pick_source_utterances_(uid)

        # load audios
        speech_length = data[self.mixture_source_name].shape[0]
        ref_audios = [self._read_source_(key, speech_length) for key in source_keys]
        ref_audios = [data[self.mixture_source_name]] + ref_audios

        # apply random gain to speech sources

        gain_in_db = [
            random.uniform(-self.dynamic_mixing_gain_db, self.dynamic_mixing_gain_db)
            for i in range(len(ref_audios))
        ]
        gain = [10 ** (g_db / 20.0) for g_db in gain_in_db]

        ref_audios = [ref * g for ref, g in zip(ref_audios, gain)]

        speech_mix = np.sum(np.array(ref_audios), axis=0)

        for i, ref in enumerate(ref_audios):
            data[f"{self.speech_ref_name_prefix}{i+1}"] = ref
        data[self.speech_name] = speech_mix

        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        # TODO(Chenda): need to test for multi-channel data.
        assert (
            len(data[self.mixture_source_name].shape) == 1
        ), "Multi-channel input has not been tested"

        # Add the category information (an integer) to `data`
        if not self.categories and "category" in data:
            raise ValueError(
                "categories must be set in the config file when utt2category files "
                "exist in the data directory (e.g., dump/raw/*/utt2category)"
            )
        if self.categories and "category" in data:
            category = data.pop("category")
            assert category in self.categories, category
            data["utt2category"] = np.array([self.categories[category]])

        if self.train:
            data = self._mix_speech_(uid, data)

        assert check_return_type(data)
        return data


class EnhPreprocessor(CommonPreprocessor):
    """Preprocessor for Speech Enhancement (Enh) task."""

    def __init__(
        self,
        train: bool,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 1,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
        channel_reordering: bool = False,
        categories: Optional[List] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        flexible_numspk: bool = False,
    ):
        super().__init__(
            train=train,
            token_type=None,
            token_list=None,
            bpemodel=None,
            text_cleaner=None,
            g2p_type=None,
            unk_symbol="<unk>",
            space_symbol="<space>",
            non_linguistic_symbols=None,
            delimiter=None,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            fs=sample_rate,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
        )
        self.speech_ref_name_prefix = speech_ref_name_prefix
        self.noise_ref_name_prefix = noise_ref_name_prefix
        self.dereverb_ref_name_prefix = dereverb_ref_name_prefix
        self.use_reverberant_ref = use_reverberant_ref
        self.num_spk = num_spk
        self.num_noise_type = num_noise_type
        self.sample_rate = sample_rate
        self.rir_scp = rir_scp
        self.noise_scp = noise_scp
        self.noise_db_range = noise_db_range
        # Whether to always convert the signals to single-channel
        self.force_single_channel = force_single_channel
        # If True, randomly reorder the channels of the multi-channel signals
        self.channel_reordering = channel_reordering

        # If specified, the audios will be chomped to the specified length
        self.speech_segment = speech_segment
        # Only used when `speech_segment` is specified.
        # If True, make sure all chomped segments are not all-zero.
        self.avoid_allzero_segment = avoid_allzero_segment

        # If True, load variable numbers of speakers in each sample, and
        # self.num_spk is regarded as the maximum possible number of speakers
        self.flexible_numspk = flexible_numspk

        # Map each category into a unique integer
        self.categories = {}
        if categories:
            count = 0
            for c in categories:
                if c not in self.categories:
                    self.categories[c] = count
                    count += 1

        if self.speech_volume_normalize is not None:
            sps = speech_volume_normalize.split("_")
            if len(sps) == 1:
                self.volume_low, self.volume_high = float(sps[0])
            elif len(sps) == 2:
                self.volume_low, self.volume_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error for --speech_volume_normalize: "
                    f"'{speech_volume_normalize}'"
                )

        if (self.rirs is not None and self.rir_apply_prob > 0) or (
            self.noises is not None and self.noise_apply_prob > 0
        ):
            logging.warning(
                "Note: Please ensure the sampling rates of all data, including audios "
                f"and RIRs, are all equal to {self.sample_rate} Hz when applying "
                "dynamic mixing."
            )

    def __basic_str__(self):
        msg = f", num_spk={self.num_spk}"
        for key in (
            "force_single_channel",
            "channel_reordering",
            "speech_volume_normalize",
        ):
            if getattr(self, key):
                msg += f", {key}={getattr(self, key)}"
        if self.rirs is not None and self.rir_apply_prob > 0:
            msg += f", sample_rate={self.sample_rate}"
            msg += f", rir_scp={self.rir_scp}, rir_apply_prob={self.rir_apply_prob}"
            if self.use_reverberant_ref:
                msg += f", use_reverberant_ref={self.use_reverberant_ref}"
        if self.noises is not None and self.noise_apply_prob > 0:
            msg += f", noise_scp={self.noise_scp}"
            msg += f", noise_apply_prob={self.noise_apply_prob}"
            msg += f", noise_db_range={self.noise_db_range}"
        if self.data_aug and self.data_aug_prob > 0:
            msg += f", data_aug={self.data_aug}, data_aug_prob={self.data_aug_prob}"
        if self.speech_segment:
            msg += f", speech_segment={self.speech_segment}"
            msg += f", avoid_allzero_segment={self.avoid_allzero_segment}"
        if self.flexible_numspk:
            msg += f", flexible_numspk={self.flexible_numspk}"
        if self.categories:
            if len(self.categories) <= 10:
                msg += f", categories={self.categories}"
            else:
                msg += f", num_category={len(self.categories)}"
        return msg

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        msg = f"{name}(train={self.train}"
        msg += self.__basic_str__()
        return msg + ")"

    def _ensure_2d(self, signal):
        if isinstance(signal, tuple):
            return tuple(self._ensure_2d(sig) for sig in signal)
        elif isinstance(signal, list):
            return [self._ensure_2d(sig) for sig in signal]
        else:
            # (Nmic, Time)
            return signal[None, :] if signal.ndim == 1 else signal.T

    def _get_early_signal(self, speech, rir, power):
        predelay = 50  # milliseconds
        dt = np.argmax(rir, axis=1).min()
        et = dt + (predelay * self.sample_rate) // 1000
        rir_early = rir[:, :et]
        speech2 = scipy.signal.convolve(speech, rir_early, mode="full")[
            :, : speech.shape[1]
        ]
        # Reverse mean power to the original power
        power2 = (speech2[detect_non_silence(speech2)] ** 2).mean()
        speech2 = np.sqrt(power / max(power2, 1e-10)) * speech2
        return speech2

    def _apply_to_all_signals(self, data_dict, func, num_spk):
        data_dict[self.speech_name] = func(data_dict[self.speech_name])

        for n in range(self.num_noise_type):
            noise_name = self.noise_ref_name_prefix + str(n + 1)
            if noise_name in data_dict:
                data_dict[noise_name] = func(data_dict[noise_name])

        for spk in range(num_spk):
            speech_ref_name = self.speech_ref_name_prefix + str(spk + 1)
            if self.train or speech_ref_name in data_dict:
                data_dict[speech_ref_name] = func(data_dict[speech_ref_name])

            dereverb_ref_name = self.dereverb_ref_name_prefix + str(spk + 1)
            if dereverb_ref_name in data_dict:
                data_dict[dereverb_ref_name] = func(data_dict[dereverb_ref_name])

    def _random_crop_range(
        self, data_dict, num_spk, tgt_length, uid=None, max_trials=10
    ):
        # Randomly crop the signals to the length `tgt_length`
        assert tgt_length > 0, tgt_length
        speech_refs = [
            data_dict[self.speech_ref_name_prefix + str(spk + 1)]
            for spk in range(num_spk)
        ]
        length = speech_refs[0].shape[0]
        if length <= tgt_length:
            if length < tgt_length:
                logging.warning(
                    f"The sample ({uid}) is not cropped due to its short length "
                    f"({length} < {tgt_length})."
                )
            return 0, length

        start = np.random.randint(0, length - tgt_length)
        count = 1
        if self.avoid_allzero_segment:
            # try to find a segment region that ensures all references are non-allzero
            while any_allzero([sf[start : start + tgt_length] for sf in speech_refs]):
                count += 1
                if count > max_trials:
                    logging.warning(
                        f"Can't find non-allzero segments for all references in {uid}."
                    )
                    break
                if start > 0:
                    start = np.random.randint(0, start)
                else:
                    break
        return start, start + tgt_length

    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        assert check_argument_types()

        if self.speech_name not in data:
            assert check_return_type(data)
            return data

        num_spk = self.num_spk

        # Add the category information (an integer) to `data`
        if not self.categories and "category" in data:
            raise ValueError(
                "categories must be set in the config file when utt2category files "
                "exist in the data directory (e.g., dump/raw/*/utt2category)"
            )

        # Add the sampling rate information (an integer) to `data`
        if "fs" in data:
            fs = int(data.pop("fs"))
            data["utt2fs"] = np.array([fs])
        else:
            fs = self.sample_rate

        sref_name = self.speech_ref_name_prefix + "1"
        if self.flexible_numspk and sref_name in data:
            # The number of speaker varies in each sample.
            # Different speaker signals are stacked in the first dimension.
            dref_name = self.dereverb_ref_name_prefix + "1"
            num_spk = len(data[sref_name])
            for i in range(2, self.num_spk + 1):
                data.pop(self.speech_ref_name_prefix + str(i), None)
                data.pop(self.dereverb_ref_name_prefix + str(i), None)
            # Divide the stacked signals into single speaker signals for consistency
            for i in range(num_spk - 1, -1, -1):
                idx = str(i + 1)
                # make sure no np.nan paddings are in the data
                assert not np.isnan(np.sum(data[sref_name][i])), uid
                data[self.speech_ref_name_prefix + idx] = data[sref_name][i]
                if dref_name in data:
                    # make sure no np.nan paddings are in the data
                    assert not np.isnan(np.sum(data[dref_name][i])), uid
                    data[self.dereverb_ref_name_prefix + idx] = data[dref_name][i]

        if self.train:
            if self.speech_segment is not None:
                speech_segment = self.speech_segment // self.sample_rate * fs
                start, end = self._random_crop_range(
                    data, num_spk, speech_segment, uid=uid
                )
                self._apply_to_all_signals(data, lambda x: x[start:end], num_spk)
            # clean speech signal (Nmic, Time)
            speech_ref = [
                self._ensure_2d(data[self.speech_ref_name_prefix + str(i + 1)])
                for i in range(num_spk)
            ]

            # dereverberated (noisy) signal (Nmic, Time)
            if self.dereverb_ref_name_prefix + "1" in data:
                dereverb_speech_ref = [
                    self._ensure_2d(data[self.dereverb_ref_name_prefix + str(i + 1)])
                    for i in range(num_spk)
                    if self.dereverb_ref_name_prefix + str(i + 1) in data
                ]
                assert len(dereverb_speech_ref) in (1, num_spk), len(
                    dereverb_speech_ref
                )
            else:
                dereverb_speech_ref = None

            # Calc power on non silence region
            power_ref = [
                (sref[detect_non_silence(sref)] ** 2).mean() for sref in speech_ref
            ]

            speech_mix = self._ensure_2d(data[self.speech_name])
            # 1. Convolve RIR
            if self.rirs is not None and self.rir_apply_prob >= np.random.random():
                speech_ref, rir_ref = zip(
                    *[
                        self._convolve_rir(
                            sp,
                            power,
                            self.rirs,
                            tgt_fs=fs,
                            single_channel=self.force_single_channel,
                        )
                        for sp, power in zip(speech_ref, power_ref)
                    ]
                )
                if self.force_single_channel:
                    speech_ref = list(map(lambda x: x[:1], speech_ref))
                    rir_ref = list(map(lambda x: x[:1], rir_ref))

                if self.use_reverberant_ref:
                    for spk in range(num_spk):
                        suffix = str(spk + 1)
                        speech_ref_name = self.speech_ref_name_prefix + suffix
                        # (Time, Nmic)
                        data[speech_ref_name] = speech_ref[spk].T

                        if dereverb_speech_ref is not None:
                            if spk == 0 or len(dereverb_speech_ref) > 1:
                                dereverb_name = self.dereverb_ref_name_prefix + suffix
                                data[dereverb_name] = self._get_early_signal(
                                    speech_ref[spk], rir_ref[spk], power_ref[spk]
                                ).T
                else:
                    for spk in range(num_spk):
                        suffix = str(spk + 1)
                        speech_ref_name = self.speech_ref_name_prefix + suffix
                        # clean speech with early reflections (Time, Nmic)
                        data[speech_ref_name] = self._get_early_signal(
                            speech_ref[spk], rir_ref[spk], power_ref[spk]
                        ).T

                        if dereverb_speech_ref is not None:
                            if spk == 0 or len(dereverb_speech_ref) > 1:
                                dereverb_name = self.dereverb_ref_name_prefix + suffix
                                data[dereverb_name] = data[speech_ref_name]

                if self.noise_ref_name_prefix + "1" in data:
                    noise = data[self.noise_ref_name_prefix + "1"]
                    speech_mix = sum(speech_ref) + noise
                else:
                    speech_mix = sum(speech_ref)

                # Add category information for dynamic mixing
                # "_reverb" means dereverberation is required
                # "_both" means both reverberant and dereverberated signals are required
                if "category" in data:
                    if self.use_reverberant_ref:
                        if dereverb_speech_ref is None:
                            if data["category"].endswith("_reverb"):
                                data["category"] = data["category"][:-7]
                            if data["category"].endswith("_both"):
                                data["category"] = data["category"][:-5]
                        else:
                            if not data["category"].endswith("_both"):
                                data["category"] = data["category"] + "_both"
                    elif not data["category"].endswith("_reverb"):
                        data["category"] = data["category"] + "_reverb"

            # 2. Add Noise
            if self.noises is not None and self.noise_apply_prob >= np.random.random():
                speech_mix = sum(speech_ref)
                if self.force_single_channel and speech_mix.shape[0] > 1:
                    speech_mix = speech_mix[:1]

                power_mix = (speech_mix[detect_non_silence(speech_mix)] ** 2).mean()
                speech_mix, noise = self._add_noise(
                    speech_mix,
                    power_mix,
                    self.noises,
                    self.noise_db_low,
                    self.noise_db_high,
                    tgt_fs=fs,
                    single_channel=self.force_single_channel,
                )

                name = self.noise_ref_name_prefix + "1"
                if name in data:
                    data[name] = noise.T
                for n in range(1, self.num_noise_type):
                    name = self.noise_ref_name_prefix + str(n + 1)
                    data.pop(name, None)

            if self.data_aug:
                if self.data_aug_prob > 0 and self.data_aug_prob >= np.random.random():
                    # Currently, we only apply data augmentation to the mixture.
                    # So, some effects should not be used for Enh, such as pitch_shift,
                    # speed_perturb, time_stretch, polarity_inverse, reverse, etc.
                    speech_mix = self.data_aug(
                        speech_mix.T if speech_mix.shape[0] > 1 else speech_mix[0],
                        self.sample_rate,
                    )

            data[self.speech_name] = speech_mix.T
            ma = np.max(np.abs(data[self.speech_name]))
            if ma > 1.0:
                self._apply_to_all_signals(data, lambda x: x / ma, num_spk)

            self._apply_to_all_signals(data, lambda x: x.squeeze(), num_spk)

        if self.force_single_channel:
            self._apply_to_all_signals(
                data, lambda x: x if x.ndim == 1 else x[:, 0], num_spk
            )

        if self.speech_volume_normalize is not None:
            if self.train:
                volume_scale = np.random.uniform(self.volume_low, self.volume_high)
            else:
                # use a fixed scale to make it deterministic
                volume_scale = self.volume_low
            ma = np.max(np.abs(data[self.speech_name]))
            self._apply_to_all_signals(data, lambda x: x * volume_scale / ma, num_spk)

        if self.categories and "category" in data:
            category = data.pop("category")
            if not re.fullmatch(r"\d+ch.*", category):
                speech_mix = data[self.speech_name]
                nch = 1 if speech_mix.ndim == 1 else speech_mix.shape[-1]
                category = f"{nch}ch_" + category
            assert category in self.categories, category
            data["utt2category"] = np.array([self.categories[category]])

        speech_mix = data[self.speech_name]
        # Reorder channels of the multi-channel signals
        if speech_mix.ndim > 1 and self.channel_reordering and self.train:
            num_ch = speech_mix.shape[-1]
            # chs = np.random.choice(range(num_ch), size=num_ch, replace=False).tolist()
            chs = np.random.permutation(num_ch).tolist()
            data[self.speech_name] = speech_mix[..., chs]
            for i in range(num_spk):
                k = self.speech_ref_name_prefix + str(i + 1)
                if self.train:
                    assert k in data, (data.keys(), k)
                if k in data and data[k].ndim > 1:
                    assert data[k].shape == speech_mix.shape
                    data[k] = data[k][..., chs]

        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(uid, data)
        data = self._text_process(data)
        return data


class SVSPreprocessor(AbsPreprocessor):
    """Preprocessor for Sing Voice Sythesis (SVS) task."""

    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        singing_volume_normalize: float = None,
        singing_name: str = "singing",
        text_name: str = "text",
        label_name: str = "label",
        midi_name: str = "score",
        fs: np.int32 = 0,
        hop_length: np.int32 = 256,
        phn_seg: dict = {
            1: [1],
            2: [0.25, 1],
            3: [0.1, 0.5, 1],
            4: [0.05, 0.1, 0.5, 1],
        },
    ):
        super().__init__(train)
        self.train = train
        self.singing_name = singing_name
        self.text_name = text_name
        self.label_name = label_name
        self.midi_name = midi_name
        self.fs = fs
        self.hop_length = hop_length
        self.singing_volume_normalize = singing_volume_normalize
        self.phn_seg = phn_seg
        self.time_shift = hop_length / fs
        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

    def __call__(
        self,
        uid: str,
        data: Dict[str, Union[str, np.ndarray, tuple]],
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        if self.singing_name in data:
            if self.singing_volume_normalize is not None:
                singing = data[self.singing_name]
                ma = np.max(np.abs(singing))
                data[self.singing_name] = singing * self.singing_volume_normalize / ma

        if self.midi_name in data and self.label_name in data:
            # Load label info
            lab_timeseq, text = data[self.label_name]
            lab_len = len(text)
            text = " ".join(text)
            text = self.text_cleaner(text)
            text = text.split(" ")
            text_ints = self.token_id_converter.tokens2ids(text)
            data.pop(self.label_name)

            label = np.zeros((lab_len))
            midi = np.zeros((lab_len))
            duration_phn = np.zeros((lab_len))
            duration_ruled_phn = np.zeros((lab_len))
            duration_syb = np.zeros((lab_len))
            slur = np.zeros((lab_len))
            # Load score info
            tempo, syb_info = data[self.midi_name]
            phn_cnt = []

            # Calculate features
            index_lab = 0

            for st, et, syb, note, phns in syb_info:
                dur = et - st
                _duration_syb = int(dur / self.time_shift + 0.5)
                phone = phns.split("_")
                phn_num = len(phone)
                phn_cnt.append(phn_num)
                pre_seg = 0
                for k in range(phn_num):
                    _duration_ruled_phn = int(
                        (self.phn_seg[phn_num][k] - pre_seg) * dur / self.time_shift
                        + 0.5
                    )
                    pre_seg = self.phn_seg[phn_num][k]
                    # timeseq from lab
                    assert text[index_lab] == phone[k]
                    _duration_phn = int(
                        (lab_timeseq[index_lab][1] - lab_timeseq[index_lab][0])
                        / self.time_shift
                        + 0.5
                    )
                    # phone level feature
                    label[index_lab] = text_ints[index_lab]
                    midi[index_lab] = note
                    duration_phn[index_lab] = _duration_phn
                    duration_ruled_phn[index_lab] = _duration_ruled_phn
                    duration_syb[index_lab] = _duration_syb
                    if syb == "—":
                        slur[index_lab] = 1
                    else:
                        slur[index_lab] = 0
                    index_lab += 1

            assert index_lab == lab_len
            data.pop(self.midi_name)

            phn_cnt = np.array(phn_cnt)
            label = label.astype(np.int64)
            midi = midi.astype(np.int64)
            duration_phn = duration_phn.astype(np.int64)
            duration_syb = duration_syb.astype(np.int64)
            duration_ruled_phn = duration_ruled_phn.astype(np.int64)
            phn_cnt = phn_cnt.astype(np.int64)
            slur = slur.astype(np.int64)

            data["label"] = label
            data["midi"] = midi
            data["duration_phn"] = duration_phn
            data["duration_ruled_phn"] = duration_ruled_phn
            data["duration_syb"] = duration_syb
            data["phn_cnt"] = phn_cnt
            data["slur"] = slur

        # TODO(Yuning): Add score from midi

        if self.text_name in data and self.tokenizer is not None:
            # FIX ME (Yuning): wrong transfer happen in pyopenjtalk
            text = data[self.text_name]
            if not isinstance(text, np.ndarray):
                if not isinstance(text, str):
                    text = " ".join(text)
                text = self.text_cleaner(text)
                tokens = self.tokenizer.text2tokens(text)
                _text_ints = self.token_id_converter.tokens2ids(tokens)
                data[self.text_name] = np.array(_text_ints, dtype=np.int64)

        return data


class TSEPreprocessor(EnhPreprocessor):
    """Preprocessor for Target Speaker Extraction."""

    def __init__(
        self,
        train: bool,
        train_spk2enroll: str = None,
        enroll_segment: int = None,
        load_spk_embedding: bool = False,
        load_all_speakers: bool = False,
        # inherited from EnhPreprocessor
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 1,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
        channel_reordering: bool = False,
        categories: Optional[List] = None,
        data_aug_effects: List = None,
        data_aug_num: List[int] = [1, 1],
        data_aug_prob: float = 0.0,
        speech_segment: Optional[int] = None,
        avoid_allzero_segment: bool = True,
        flexible_numspk: bool = False,
    ):
        super().__init__(
            train,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            speech_ref_name_prefix=speech_ref_name_prefix,
            noise_ref_name_prefix=noise_ref_name_prefix,
            dereverb_ref_name_prefix=dereverb_ref_name_prefix,
            use_reverberant_ref=use_reverberant_ref,
            num_spk=num_spk,
            num_noise_type=num_noise_type,
            sample_rate=sample_rate,
            force_single_channel=force_single_channel,
            channel_reordering=channel_reordering,
            categories=categories,
            data_aug_effects=data_aug_effects,
            data_aug_num=data_aug_num,
            data_aug_prob=data_aug_prob,
            speech_segment=speech_segment,
            avoid_allzero_segment=avoid_allzero_segment,
            flexible_numspk=flexible_numspk,
        )
        # If specified, the enrollment will be chomped to the specified length
        self.enroll_segment = enroll_segment
        # If True, the speaker embedding will be loaded instead of enrollment audios
        self.load_spk_embedding = load_spk_embedding
        # If False, only one of the speakers in each mixture sample will be loaded
        self.load_all_speakers = load_all_speakers

        if train and rir_scp is not None and rir_apply_prob > 0:
            logging.warning(
                "Be cautious when applying RIRs on the fly in the TSE task! "
                "Please ensure `speech_ref` sums up to `speech_mix` for each sample."
            )

        if train:
            if train_spk2enroll is None:
                logging.info("Using fixed enrollment for each sample")
                self.train_spk2enroll = None
            else:
                logging.info("Using dynamically sampled enrollment for each sample")
                with open(train_spk2enroll, "r", encoding="utf-8") as f:
                    # {spkID: [(uid1, path1), (uid2, path2), ...]}
                    self.train_spk2enroll = json.load(f)
        else:
            self.train_spk2enroll = None

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        msg = f"{name}(train={self.train}"
        if self.train_spk2enroll:
            msg += f", len(train_spk2enroll)={len(self.train_spk2enroll)}"
        for key in ("enroll_segment", "load_spk_embedding", "load_all_speakers"):
            if getattr(self, key):
                msg += f", {key}={getattr(self, key)}"
        msg += self.__basic_str__()
        return msg + ")"

    def _read_audio_segment(self, path, seg_len=None):
        with soundfile.SoundFile(path) as f:
            if seg_len is None or f.frames == seg_len:
                audio = f.read(dtype=np.float32, always_2d=True)
            elif f.frames < seg_len:
                offset = np.random.randint(0, seg_len - f.frames)
                # audio: (Time, Nmic)
                audio = f.read(dtype=np.float32, always_2d=True)
                # Repeat audio
                audio = np.pad(
                    audio,
                    [(offset, seg_len - f.frames - offset), (0, 0)],
                    mode="wrap",
                )
            else:
                offset = np.random.randint(0, f.frames - seg_len)
                f.seek(offset)
                # audio: (Time, Nmic)
                audio = f.read(seg_len, dtype=np.float32, always_2d=True)
            if len(audio) != seg_len:
                raise RuntimeError(f"Something wrong: {path}")
        return audio[:, 0]

    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        assert check_argument_types()

        ref_names = [k for k in data.keys() if re.match(r"speech_ref\d+", k)]
        num_spk = len(ref_names)

        aux_names = [k for k in data.keys() if re.match(r"enroll_ref\d+", k)]
        if self.flexible_numspk:
            # The number of speaker varies in each sample.
            # Different speaker signals are stacked in the first dimension.
            enroll_name = "enroll_ref1"
            for name in aux_names:
                if name != enroll_name:
                    data.pop(name)
            aux_names = [f"enroll_ref{i + 1}" for i in range(num_spk)]
            # Divide the concatenated enrollments into single speaker enrollments
            # NOTE(wangyou): whitespace is not allowed inside each path
            tup = data[enroll_name].split()
            if len(tup) == num_spk:
                # normal format in `enroll_spk1.scp`:
                # MIXTURE_UID /path/to/enrollment_or_embedding
                for i in range(num_spk - 1, -1, -1):
                    data[f"enroll_ref{i + 1}"] = tup[i]
            elif len(tup) == num_spk * 2:
                # a special format in `enroll_spk1.scp`:
                # MIXTURE_UID *UID SPEAKER_ID
                for i in range(num_spk - 1, -1, -1):
                    data[f"enroll_ref{i + 1}"] = " ".join(tup[i * 2 : i * 2 + 2])
            else:
                raise ValueError(
                    f"Invalid format with in enroll_spk1.scp. Expected {num_spk} or "
                    f"{num_spk * 2} columns, got {len(tup)} columns:\n{tup}"
                )

        if self.train:
            assert len(ref_names) == len(aux_names), (len(ref_names), len(aux_names))
            if not self.load_all_speakers:
                # only load one target-speaker data
                spk = np.random.randint(0, num_spk)
                for i, name in enumerate(ref_names):
                    if i == 0:
                        data[name] = data[ref_names[spk]]
                    else:
                        data.pop(name)
                        continue

            for i, name in enumerate(aux_names):
                if not self.load_all_speakers:
                    if i == 0:
                        data[name] = data[aux_names[spk]]
                    else:
                        data.pop(name)
                        continue
                if self.train_spk2enroll is None:
                    # normal format in `enroll_spk?.scp`:
                    # MIXTURE_UID /path/to/enrollment_or_embedding
                    assert not data[name].startswith("*"), data[name]
                    aux_audio = data[name]
                else:
                    # a special format in `enroll_spk?.scp`:
                    # MIXTURE_UID *UID SPEAKER_ID
                    assert data[name].startswith("*"), data[name]
                    cur_uid, spkid = data[name][1:].strip().split(maxsplit=1)
                    aux_uid, aux_audio = random.choice(self.train_spk2enroll[spkid])
                    while aux_uid == cur_uid:
                        aux_uid, aux_audio = random.choice(self.train_spk2enroll[spkid])
                if getattr(self, "load_spk_embedding", False):
                    data[name] = np.load(aux_audio)[None, :]  # force 2D
                elif self.enroll_segment:
                    data[name] = self._read_audio_segment(
                        aux_audio, self.enroll_segment
                    )
                else:
                    data[name] = soundfile.read(aux_audio)[0]
        else:
            for name in aux_names:
                if data[name].startswith("*"):
                    # in case of collecting stats for training data
                    data[name] = np.zeros(1, dtype=data[self.speech_name].dtype)
                else:
                    if getattr(self, "load_spk_embedding", False):
                        data[name] = np.load(data[name])[None, :]  # force 2D
                    elif self.enroll_segment and len(ref_names) > 0:
                        # do not segment the enrollment during inference
                        # (stage 7 in enh.sh)
                        data[name] = self._read_audio_segment(
                            data[name], self.enroll_segment
                        )
                    else:
                        data[name] = soundfile.read(data[name])[0]

        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = super()._speech_process(uid, data)
        data = self._speech_process(uid, data)
        return data


class SpkPreprocessor(CommonPreprocessor):
    """Preprocessor for Speaker tasks.

    Args:
        train (bool): Whether to use in training mode.
        spk2utt (str): Path to the `spk2utt` file.
        target_duration (float): Target duration in seconds.
        sample_rate (int): Sampling rate.
        num_eval (int): Number of utterances to be used for evaluation.
        rir_scp (str): Path to the RIR scp file.
        rir_apply_prob (float): Probability of applying RIR.
        noise_info (List[Tuple[float, str, Tuple[int, int], Tuple[float, float]]]):
            List of tuples of noise information. Each tuple represents a noise type.
            Each tuple consists of `(prob, noise_scp, num_to_mix, db_range)`.
                - `prob` (float) is the probability of applying the noise type.
                - `noise_scp` (str) is the path to the noise scp file.
                - `num_to_mix` (Tuple[int, int]) is the range of the number of noises
                    to be mixed.
                - `db_range` (Tuple[float, float]) is the range of noise levels in dB.
        noise_apply_prob (float): Probability of applying noise.
        short_noise_thres (float): Threshold of short noise.
    """

    def __init__(
        self,
        train: bool,
        target_duration: float,  # in seconds
        spk2utt: str = None,
        sample_rate: int = 16000,
        num_eval: int = 10,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_info: List[
            Tuple[float, str, Tuple[int, int], Tuple[float, float]]
        ] = None,
        noise_apply_prob: float = 1.0,
        short_noise_thres: float = 0.5,
    ):
        super().__init__(train, rir_scp=rir_scp, rir_apply_prob=rir_apply_prob)

        self.spk2label = None  # a dictionary that maps string speaker label to int
        self.sample_rate = sample_rate
        self.target_duration = int(target_duration * sample_rate)
        self.num_eval = num_eval

        if train:
            with open(spk2utt, "r") as f_s2u:
                self.spk2utt = f_s2u.readlines()
            self._make_label_mapping()
            self.nspk = len(self.spk2utt)

        self.rir_scp = rir_scp

        self.noise_apply_prob = noise_apply_prob
        self.short_noise_thres = short_noise_thres
        self.noises = []
        self.noise_probs = []
        self.noise_db_ranges = []
        self.noise_num_to_mix = []
        if noise_apply_prob > 0:
            for prob, noise_scp, num_to_mix, db_range in noise_info:
                if prob > 0:
                    assert len(db_range) == 2, db_range
                    assert db_range[0] <= db_range[1], db_range
                    assert len(num_to_mix) == 2, num_to_mix
                    assert num_to_mix[0] <= num_to_mix[1], num_to_mix
                    self.noise_probs.append(prob)
                    self.noise_db_ranges.append(tuple(db_range))
                    self.noise_num_to_mix.append(num_to_mix)
                    noises = []
                    with open(noise_scp, "r", encoding="utf-8") as f:
                        for line in f:
                            sps = line.strip().split(None, 1)
                            if len(sps) == 1:
                                noises.append(sps[0])
                            else:
                                noises.append(sps[1])
                    self.noises.append(noises)

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        msg = f"{name}(train={self.train}"
        if self.spk2label:
            msg += f", len(spk2label)={len(self.spk2label)}"
        for key in ("target_duration", "sample_rate", "num_eval"):
            if getattr(self, key):
                msg += f", {key}={getattr(self, key)}"
        if self.rirs is not None and self.rir_apply_prob > 0:
            msg += f", rir_scp={self.rir_scp}, rir_apply_prob={self.rir_apply_prob}"
        if self.noise_apply_prob > 0 and self.noises:
            msg += f", noise_apply_prob={self.noise_apply_prob}"
            msg += f", noises.shapes={[len(n) for n in self.noises]}"
            msg += f", noise_probs={self.noise_probs}"
            msg += f", noise_db_ranges={self.noise_db_ranges}"
            msg += f", noise_num_to_mix={self.noise_num_to_mix}"
        return msg + ")"

    def _make_label_mapping(self):
        label_idx = 0
        self.spk2label = {}
        for spk in self.spk2utt:
            spk = spk.strip().split(" ")[0]
            self.spk2label[spk] = label_idx
            label_idx += 1

    def _speech_process(self, data: Dict[np.ndarray, str]):
        if self.train:
            audio = data["speech"]

            # duplicate if utt is shorter than minimum required duration
            if len(audio) < self.target_duration:
                shortage = self.target_duration - len(audio) + 1
                audio = np.pad(audio, (0, shortage), "wrap")

            startframe = np.array(
                [np.int64(random.random() * (len(audio) - self.target_duration))]
            )

            data["speech"] = audio[
                int(startframe) : int(startframe) + self.target_duration
            ]

            if self.noise_apply_prob > 0 or self.rir_apply_prob > 0:
                data["speech"] = self._apply_data_augmentation(data["speech"])
        else:
            audio = data["speech"]
            audio2 = data["speech2"]

            # duplicate if utt is shorter than minimum required duration
            if len(audio) < self.target_duration:
                shortage = self.target_duration - len(audio) + 1
                audio = np.pad(audio, (0, shortage), "wrap")
            if len(audio2) < self.target_duration:
                shortage = self.target_duration - len(audio2) + 1
                audio2 = np.pad(audio2, (0, shortage), "wrap")

            startframe = np.linspace(
                0, len(audio) - self.target_duration, num=self.num_eval
            )
            audios = []
            for frame in startframe:
                audios.append(audio[int(frame) : int(frame) + self.target_duration])
            audios = np.stack(audios, axis=0)

            startframe2 = np.linspace(
                0, len(audio2) - self.target_duration, num=self.num_eval
            )
            audios2 = []
            for frame in startframe2:
                audios2.append(audio2[int(frame) : int(frame) + self.target_duration])
            audios2 = np.stack(audios2, axis=0)

            data["speech"] = audios
            data["speech2"] = audios2

        return data

    def _convolve_rir(self, speech, rirs):
        rir_path = np.random.choice(rirs)
        rir = None
        if rir_path is not None:
            rir, _ = soundfile.read(rir_path, dtype=np.float64, always_2d=True)

            # rir: (Nmic, Time)
            rir = rir.T

            # normalize rir
            rir = rir / np.sqrt(np.sum(rir**2))

            # speech: (Nmic, Time)
            # Note that this operation doesn't change the signal length
            speech = scipy.signal.convolve(speech, rir, mode="full")[
                :, : speech.shape[1]
            ]
        return speech, rir

    def _load_noise(self, speech, speech_db, noises, noise_db_low, noise_db_high):
        nsamples = speech.shape[1]
        noise_path = np.random.choice(noises)
        noise = None
        if noise_path is not None:
            noise_snr = np.random.uniform(noise_db_low, noise_db_high)
            with soundfile.SoundFile(noise_path) as f:
                if f.frames == nsamples:
                    noise = f.read(dtype=np.float64)
                elif f.frames < nsamples:
                    # noise: (Time,)
                    noise = f.read(dtype=np.float64)
                    # Repeat noise
                    noise = np.pad(
                        noise,
                        (0, nsamples - f.frames),
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples)
                    f.seek(offset)
                    # noise: (Time,)
                    noise = f.read(nsamples, dtype=np.float64)
                    if len(noise) != nsamples:
                        raise RuntimeError(f"Something wrong: {noise_path}")
            # noise: (Nmic, Time)
            noise = noise[None, :]

            noise_power = np.mean(noise**2)
            noise_db = 10 * np.log10(noise_power + 1e-4)
            scale = np.sqrt(10 ** ((speech_db - noise_db - noise_snr) / 10))

            noise = noise * scale
        return noise

    def _apply_data_augmentation(self, speech):
        # speech: (Nmic, Time)
        if speech.ndim == 1:
            speech = speech[None, :]
        else:
            speech = speech.T

        if self.rirs is not None and self.rir_apply_prob >= np.random.random():
            speech, _ = self._convolve_rir(speech, self.rirs)

        if self.noises and self.noise_apply_prob >= np.random.random():
            idx = random.choices(
                range(len(self.noises)), weights=self.noise_probs, k=1
            )[0]
            low, high = self.noise_num_to_mix[idx]
            if low == high:
                num_to_mix = low
            else:
                num_to_mix = np.random.randint(low, high + 1)

            # add eps of 1e-4 to avoid negative value before log
            speech_db = 10 * np.log10(np.mean(speech**2) + 1e-4)
            noiselist = []
            for _ in range(num_to_mix):
                noise = self._load_noise(
                    speech,  # original speech
                    speech_db,  # db of speech
                    self.noises[idx],  # a list of a type of noise
                    self.noise_db_ranges[idx][0],  # min db
                    self.noise_db_ranges[idx][1],  # max db
                )
                noiselist.append(noise)
            noise = np.sum(np.concatenate(noiselist, axis=0), axis=0, keepdims=True)
            speech = speech + noise

        speech = np.squeeze(speech, axis=0)
        return speech

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Make speaker labels into integers."""
        if self.train:
            int_label = self.spk2label[data["spk_labels"]]
            data["spk_labels"] = np.asarray([int_label], dtype=np.int64)
        else:
            data["spk_labels"] = np.asarray([int(data["spk_labels"])])

        if "task_tokens" in data:
            data["task_tokens"] = np.asarray([int(data["task_tokens"])])

        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._text_process(data)
        data = self._speech_process(data)

        return data


class S2TPreprocessor(CommonPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        text_prev_name: str = "text_prev",
        text_ctc_name: str = "text_ctc",
        fs: int = 16000,
        na_symbol: str = "<na>",  # text is not available e.g. for prev or ctc
        speech_length: float = 30,  # pad or trim speech to this value in seconds
        speech_resolution: float = 0.02,  # speech time resolution
        speech_init_silence: float = 1.0,  # max silence before speech for data aug
        text_prev_apply_prob: float = 0.5,  # whether to condition on text_prev
        time_apply_prob: float = 0.5,  # whether to include timestamps
        notime_symbol: str = "<notimestamps>",
        first_time_symbol: str = "<0.00>",
        last_time_symbol: str = "<30.00>",
    ):
        super().__init__(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            text_name=text_name,
            fs=fs,
        )
        self.text_prev_name = text_prev_name
        self.text_ctc_name = text_ctc_name
        self.speech_length = int(speech_length * fs)
        self.speech_resolution = int(speech_resolution * fs)
        self.speech_init_silence = int(speech_init_silence * fs)
        self.text_prev_apply_prob = text_prev_apply_prob
        self.time_apply_prob = time_apply_prob

        # Obtain the token id of special tokens
        self.na_symbol = na_symbol
        self.notime = self.token_id_converter.token2id[notime_symbol]
        self.first_time = self.token_id_converter.token2id[first_time_symbol]
        self.last_time = self.token_id_converter.token2id[last_time_symbol]

    def _pad_or_trim_speech(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Tuple[Dict[str, Union[str, np.ndarray]], int]:
        assert check_argument_types()

        init_pad = 0
        if self.speech_name in data:
            speech = data[self.speech_name]

            # speech: (Nmic, Time)
            if speech.ndim == 1:
                speech = speech[None, :]
            else:
                speech = speech.T

            # Add silence to the left
            if self.train and speech.shape[-1] < self.speech_length:
                init_pad = np.random.randint(
                    min(self.speech_length - speech.shape[-1], self.speech_init_silence)
                    + 1
                )
                speech = np.pad(speech, ((0, 0), (init_pad, 0)))

            # Pad or trim to max_samples
            if speech.shape[-1] < self.speech_length:
                speech = np.pad(
                    speech, ((0, 0), (0, self.speech_length - speech.shape[-1]))
                )
            else:
                speech = speech[:, : self.speech_length]

            data[self.speech_name] = speech.T  # convert back to time first

        assert check_return_type((data, init_pad))
        return data, init_pad

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]], time_shift: int
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        text_names = [self.text_name, self.text_prev_name, self.text_ctc_name]
        if self.tokenizer is not None:
            for name in text_names:
                if name in data:
                    text = data[name]

                    # Remove prev text by setting it to <na>
                    if (
                        self.train
                        and name == self.text_prev_name
                        and np.random.uniform() > self.text_prev_apply_prob
                    ):
                        text = self.na_symbol

                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.text2tokens(text)
                    text_ints = self.token_id_converter.tokens2ids(tokens)
                    text_ints = np.array(text_ints, dtype=np.int64)

                    # Augment text
                    if name == self.text_name:
                        # NOTE(yifan): The first token is always space
                        # which should be removed.
                        # No space is allowed between special tokens.
                        # This works for bpe, but maybe not for the other types.
                        text_ints = text_ints[1:]

                        # Remove timestamps
                        if self.train and np.random.uniform() > self.time_apply_prob:
                            # Timestamps are continuous ints
                            text_ints = text_ints[
                                np.logical_or(
                                    text_ints < self.first_time,
                                    text_ints > self.last_time,
                                )
                            ]
                            # First two tokens are <category> and <task>
                            text_ints = np.insert(text_ints, 2, self.notime)

                        # Shift timestamps
                        text_ints[
                            np.logical_and(
                                text_ints >= self.first_time,
                                text_ints <= self.last_time,
                            )
                        ] += time_shift

                    data[name] = text_ints

        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data, init_pad = self._pad_or_trim_speech(data)

        data = self._text_process(data, round(init_pad / self.speech_resolution))

        return data
