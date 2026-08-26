"""F5-TTS inference engine.

Built by a recipe's ``infer`` stage
(``model._target_: espnet3.systems.tts.f5_tts.inference.F5TTSInference``).
For each test sample the runner calls ``model(**{key: data[key] for key in input_key})``
with ``input_key: [text, ref_speech, ref_text]`` (cross/same-speaker protocol) and
feeds the result to ``src.inference.build_output`` (which needs a ``"wav"`` entry).

The model is rebuilt from the *training* config by instantiating that config's
own ``model`` block, so it stays in sync with whatever was trained:
``ESPnetTTSModel(feats_extract=VocoderMelSpec, tts=F5TTS(...))``, assembled by
``espnet3.systems.tts.f5_tts.builder.build_f5_tts_model``. Text is tokenized with
the exact espnet2 components used in training (TextCleaner + tokenizer +
TokenIDConverter) read from the training config's preprocessor.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter

logger = logging.getLogger(__name__)


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into <= max_chars (utf-8 bytes) chunks. Ported from F5 chunk_text."""
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
    for sentence in sentences:
        if not sentence:
            continue
        if (
            len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8"))
            <= max_chars
        ):
            current_chunk += (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def _cross_fade(
    waves: List[np.ndarray], cross_fade_duration: float, sample_rate: int
) -> np.ndarray:
    """Concatenate waves with a linear cross-fade.

    Ported from F5 infer_batch_process.
    """
    if not waves:
        return np.zeros(1, dtype=np.float32)
    if len(waves) == 1:
        return waves[0]
    if cross_fade_duration <= 0:
        return np.concatenate(waves)
    final_wave = waves[0]
    for next_wave in waves[1:]:
        prev_wave = final_wave
        n = int(cross_fade_duration * sample_rate)
        n = min(n, len(prev_wave), len(next_wave))
        if n <= 0:
            final_wave = np.concatenate([prev_wave, next_wave])
            continue
        prev_overlap = prev_wave[-n:]
        next_overlap = next_wave[:n]
        fade_out = np.linspace(1, 0, n)
        fade_in = np.linspace(0, 1, n)
        cross = prev_overlap * fade_out + next_overlap * fade_in
        final_wave = np.concatenate([prev_wave[:-n], cross, next_wave[n:]])
    return final_wave


class F5TTSInference:
    """Load a trained F5-TTS model + vocoder and synthesize waveforms."""

    def __init__(
        self,
        train_config: str,
        ckpt_path: str,
        device: str = "cpu",
        use_ema: bool = True,
        vocoder_name: str = "vocos",
        vocoder_path: Optional[str] = None,
        target_sample_rate: int = 24000,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
        native_f5: bool = False,
        seed: Optional[int] = None,
    ):
        """Build the model, tokenizer and vocoder for inference.

        Args:
            train_config: Path to the training YAML (provides the ``model``
                block + preprocessor tokenization settings, single source of
                truth).
            ckpt_path: Lightning checkpoint (``.ckpt``) produced by training.
            device: Torch device string.
            use_ema: Load EMA-averaged weights (``ema_model_state_dict``) when
                present; otherwise the raw ``state_dict``.
            vocoder_name / vocoder_path: ``"vocos"`` (default) or ``"bigvgan"``.
            target_sample_rate: Output/vocoder sample rate.
            nfe_step / cfg_strength / sway_sampling_coef / speed / seed:
                Sampling hyperparameters forwarded to ``CFM.sample``.
            native_f5: Load an OFFICIAL SWivid/F5-TTS checkpoint (``.pt`` or
                ``.safetensors``) instead of an espnet/Lightning ckpt. The weights
                are loaded straight into the ported CFM (``model.tts.cfm``), so
                the architecture (``tts_conf``) and the pinyin ``token_list`` in
                ``train_config`` MUST match the pretrained model (F5TTS_Base +
                ``Emilia_ZH_EN_pinyin/vocab.txt``). Use this to sanity-check the
                inference + tokenization path against known-good weights.

        Raises:
            ValueError: If ``train_config`` has no ``model._target_``, if its
                ``dataset.preprocessor.token_list`` is missing, or if
                ``vocoder_name`` is neither ``"vocos"`` nor ``"bigvgan"``.
            ImportError: If the selected vocoder package is not installed.

        Example:
            .. code-block:: yaml

                inference:
                  _target_: espnet3.systems.tts.f5_tts.inference.F5TTSInference
                  train_config: ${recipe_dir}/conf/training_f5_tts_small.yaml
                  ckpt_path: ${exp_dir}/last.ckpt
                  device: cuda
                  nfe_step: 32
                  cfg_strength: 2.0

        Note:
            ``train_config`` is the recipe's own training YAML, not the
            ``config.yaml`` written into ``exp_dir``: the model is rebuilt from
            the ``model:`` block via Hydra, so architecture and tokenizer
            settings always come from one source of truth. Construction is
            eager, loading the checkpoint and the vocoder up front.
        """
        self.device = torch.device(device)
        self.target_sample_rate = target_sample_rate
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.speed = speed
        self.target_rms = target_rms
        self.cross_fade_duration = cross_fade_duration
        self.seed = seed

        cfg = OmegaConf.to_container(OmegaConf.load(train_config), resolve=True)
        fe_conf = (cfg.get("model") or {}).get("feats_extract_conf") or {}
        self.hop_length = int(fe_conf.get("hop_length", 256))
        model = self._build_model(cfg, ckpt_path, use_ema, native_f5)
        # ESPnetTTSModel components used for generation.
        self.feats_extract = model.feats_extract
        self.cfm = model.tts.cfm
        self.model = model

        self._build_tokenizer(cfg)
        self.vocoder = self._load_vocoder(vocoder_name, vocoder_path)

    # ------------------------------------------------------------------ build

    def _build_model(
        self, cfg: dict, ckpt_path: str, use_ema: bool, native_f5: bool = False
    ):
        model_cfg = cfg.get("model")
        if not model_cfg or not model_cfg.get("_target_"):
            raise ValueError(
                "train_config must set `model._target_` (the F5-TTS builder)."
            )
        logger.info("Building TTS model via %s", model_cfg["_target_"])
        model = instantiate(model_cfg)

        if native_f5:
            # Official SWivid/F5-TTS checkpoint: CFM-level keys (transformer.* /
            # mel_spec.*, EMA prefixed ema_model.). Load straight into the ported
            # CFM so the espnet wrapper prefixes (tts.cfm.) don't matter.
            cfm_sd = self._load_native_f5_state(ckpt_path, use_ema)
            missing, unexpected = model.tts.cfm.load_state_dict(cfm_sd, strict=False)
            self._log_load("F5-native -> model.tts.cfm", ckpt_path, missing, unexpected)
            return model.to(self.device).eval()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if use_ema and "ema_model_state_dict" in ckpt:
            prefix = "ema_model."
            state_dict = {
                k[len(prefix) :]: v
                for k, v in ckpt["ema_model_state_dict"].items()
                if k.startswith(prefix)
            }
            logger.info("Loading EMA weights from %s", ckpt_path)
        else:
            state_dict = ckpt.get("state_dict", ckpt)
            logger.info("Loading raw (non-EMA) weights from %s", ckpt_path)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self._log_load("espnet", ckpt_path, missing, unexpected)
        return model.to(self.device).eval()

    @staticmethod
    def _load_native_f5_state(ckpt_path: str, use_ema: bool) -> dict:
        """CFM-level state dict from an official F5-TTS checkpoint.

        Handles ``.pt`` (``torch.load`` -> ``model_state_dict`` /
        ``ema_model_state_dict``) and ``.safetensors`` (a flat EMA tensor dict).
        Returns keys at CFM level (``transformer.*``, ``mel_spec.*``): the
        ``ema_model.`` prefix is stripped and the ``initted`` / ``step``
        bookkeeping tensors are dropped, mirroring F5's own ``load_checkpoint``.
        """
        if str(ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            raw = load_file(ckpt_path)  # flat: ema_model.* (+ initted/step)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if use_ema and "ema_model_state_dict" in ckpt:
                raw = ckpt["ema_model_state_dict"]
            elif "model_state_dict" in ckpt:
                raw = ckpt["model_state_dict"]
            else:
                raw = ckpt
        return {
            k.replace("ema_model.", "", 1): v
            for k, v in raw.items()
            if k not in ("initted", "step")
        }

    @staticmethod
    def _log_load(tag: str, ckpt_path: str, missing, unexpected) -> None:
        logger.info("Loaded %s weights from %s", tag, ckpt_path)
        if missing:
            logger.warning("[%s] missing keys (%d): %s", tag, len(missing), missing)
        if unexpected:
            logger.warning(
                "[%s] unexpected keys (%d): %s", tag, len(unexpected), unexpected
            )

    def _build_tokenizer(self, cfg: dict) -> None:
        """Replicate the training-time text tokenization for inference.

        Sets ``self._tokenize_fn(text) -> int64 ids`` matching the training
        preprocessor: F5's zh+en pinyin (``F5PinyinPreprocessor``) or espnet2's
        char/phn tokenizer (``CommonPreprocessor``).
        """
        prep = cfg.get("dataset", {}).get("preprocessor", {})
        target = str(prep.get("_target_", ""))

        if "F5PinyinPreprocessor" in target or prep.get("vocab_file"):
            # F5 zh+en pinyin: F5's own tokenizer + vocab (unknown token -> 0).
            from espnet3.systems.tts.f5_tts.pinyin import (
                load_vocab_char_map,
                text_to_pinyin_ids,
            )

            vocab_char_map = load_vocab_char_map(prep["vocab_file"])
            self._tokenize_fn = lambda text: text_to_pinyin_ids(text, vocab_char_map)
            return

        # espnet2 char/phn tokenization (matches CommonPreprocessor).
        token_list = prep.get("token_list")
        if token_list is None:
            raise ValueError(
                "Could not find dataset.preprocessor.token_list in train_config."
            )
        if prep.get("g2p_type") == "f5_pinyin":
            from espnet3.systems.tts.f5_tts.pinyin import register_f5_pinyin_g2p

            register_f5_pinyin_g2p()
        cleaner = TextCleaner(prep.get("text_cleaner"))
        tokenizer = build_tokenizer(
            token_type=prep.get("token_type", "char"),
            bpemodel=prep.get("bpemodel"),
            non_linguistic_symbols=prep.get("non_linguistic_symbols"),
            g2p_type=prep.get("g2p_type"),
        )
        token_id_converter = TokenIDConverter(token_list)
        self._tokenize_fn = lambda text: np.asarray(
            token_id_converter.tokens2ids(tokenizer.text2tokens(cleaner(text))),
            dtype=np.int64,
        )

    def _load_vocoder(self, vocoder_name: str, vocoder_path: Optional[str]):
        if vocoder_name == "vocos":
            try:
                from vocos import Vocos
            except ImportError as e:
                raise ImportError(
                    "vocos is required for vocoder_name='vocos'. Install with "
                    "`pip install vocos`."
                ) from e
            if vocoder_path:
                vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
                state = torch.load(
                    f"{vocoder_path}/pytorch_model.bin", map_location="cpu"
                )
                vocoder.load_state_dict(state)
            else:
                vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        elif vocoder_name == "bigvgan":
            try:
                import bigvgan
            except ImportError as e:
                raise ImportError(
                    "bigvgan is required for vocoder_name='bigvgan'. See "
                    "https://github.com/NVIDIA/BigVGAN."
                ) from e
            repo = vocoder_path or "nvidia/bigvgan_v2_24khz_100band_256x"
            vocoder = bigvgan.BigVGAN.from_pretrained(repo, use_cuda_kernel=False)
            vocoder.remove_weight_norm()
        else:
            raise ValueError(f"Unsupported vocoder: {vocoder_name!r}.")
        return vocoder.to(self.device).eval()

    # -------------------------------------------------------------- inference

    def _tokenize(self, text: str) -> np.ndarray:
        return self._tokenize_fn(text)

    def _vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocode mel ``[1, d, n]`` to waveform ``[nw]``.

        Vocos exposes ``decode``; bigvgan is a plain ``nn.Module``.
        """
        if hasattr(self.vocoder, "decode"):
            wav = self.vocoder.decode(mel)
        else:  # bigvgan is a plain nn.Module
            wav = self.vocoder(mel)
        return wav.squeeze().detach().cpu()

    @torch.no_grad()
    def infer_one(
        self,
        gen_text: str,
        ref_audio: np.ndarray,
        ref_text: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesize ``gen_text`` in the voice of ``ref_audio``.

        Mirrors upstream F5-TTS ``infer_process``: RMS-normalize the reference,
        split the target text into reference-length-dependent chunks, sample
        each chunk, vocode, then cross-fade the pieces back together.

        Args:
            gen_text: Target text to speak.
            ref_audio: Reference waveform at ``target_sample_rate``. Multi-
                channel input is averaged down to mono.
            ref_text: Transcript of ``ref_audio``. Defaults to ``gen_text``,
                i.e. treating the reference as self-referential.

        Returns:
            Mono waveform as ``float32`` at ``target_sample_rate``. Returns a
            single zero sample when the text yields no synthesizable chunk.

        Example:
            .. code-block:: python

                >>> wav = tts.infer_one("hello world", ref_audio, ref_text="hi")
                >>> wav.dtype, wav.ndim
                (dtype('float32'), 1)

        Note:
            Output length is governed by the reference: the per-chunk duration
            is extrapolated from the reference audio-to-text ratio, so a
            mismatched ``ref_text`` skews it. The reference loudness is restored
            after vocoding, so the result matches the input level rather than
            ``target_rms``.
        """
        ref_text = gen_text if ref_text is None else ref_text
        sr = self.target_sample_rate
        hop = self.hop_length

        # Reference waveform [1, T]: mono + RMS normalization to target_rms.
        audio = torch.as_tensor(np.asarray(ref_audio), dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
        audio = audio.to(self.device)

        # F5 appends a trailing space when ref_text ends on a single-byte char.
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "

        # Reference-length-dependent chunking (same formula as infer_process).
        audio_seconds = audio.shape[-1] / sr
        max_chars = int(
            len(ref_text.encode("utf-8"))
            / audio_seconds
            * (22 - audio_seconds)
            * self.speed
        )
        gen_text_batches = _chunk_text(gen_text, max_chars=max(max_chars, 1))
        if not gen_text_batches:
            return np.zeros(1, dtype=np.float32)

        ref_audio_len = audio.shape[-1] // hop
        ref_text_len = max(len(ref_text.encode("utf-8")), 1)

        generated_waves: List[np.ndarray] = []
        for gen_chunk in gen_text_batches:
            # F5 slows down very short generations.
            local_speed = 0.3 if len(gen_chunk.encode("utf-8")) < 10 else self.speed
            gen_text_len = len(gen_chunk.encode("utf-8"))
            duration = ref_audio_len + int(
                ref_audio_len / ref_text_len * gen_text_len / local_speed
            )

            ids = (
                torch.from_numpy(self._tokenize(ref_text + gen_chunk))
                .unsqueeze(0)
                .to(self.device)
            )
            # Pass the raw reference wave as cond; CFM.sample extracts its mel
            # internally (same vocos MelSpec as feats_extract), exactly like F5.
            out, _ = self.cfm.sample(
                cond=audio,
                text=ids,
                duration=duration,
                steps=self.nfe_step,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
                seed=self.seed,
            )

            gen_mel = out[:, ref_audio_len:, :].to(torch.float32)  # drop the prompt
            if gen_mel.shape[1] == 0:
                continue
            wave = self._vocode(gen_mel.permute(0, 2, 1))  # [1, d, n_gen]
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            if rms < self.target_rms:  # de-normalize back to the reference loudness
                wave = wave * float(rms / self.target_rms)
            generated_waves.append(wave)

        if not generated_waves:
            return np.zeros(1, dtype=np.float32)
        return _cross_fade(generated_waves, self.cross_fade_duration, sr)

    def __call__(
        self,
        text: Union[str, List[str]],
        ref_speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        ref_text: Optional[Union[str, List[str]]] = None,
        speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> dict:
        """Inference entry point used by the runner.

        ``text`` is the target text. The reference audio comes from ``ref_speech``
        (cross/same-speaker protocol, with ``ref_text`` its transcript); if only
        ``speech`` is given it is used as the reference and ``ref_text`` defaults
        to ``text`` (self-reference). Supports a single sample (``batch_size:
        null``) or a list (batched).

        Args:
            text: Target text, or a list of them for a batched call.
            ref_speech: Reference waveform(s) for the cross/same-speaker
                protocol.
            ref_text: Transcript(s) of ``ref_speech``.
            speech: Fallback reference used when ``ref_speech`` is absent, which
                makes the call self-referential.

        Returns:
            ``{"wav": waveform}`` for a single sample, or ``{"wav": [...]}``
            with one waveform per input when ``text`` is a list.

        Raises:
            ValueError: If neither ``ref_speech`` nor ``speech`` is given. F5 is
                zero-shot and cannot synthesize without a reference.

        Example:
            .. code-block:: python

                >>> tts(text="hello", speech=ref_wave)["wav"].ndim
                1
                >>> len(tts(text=["a", "b"], ref_speech=[w1, w2])["wav"])
                2

        Note:
            Batched input is looped over ``infer_one`` rather than batched
            through the solver, so it costs the same as separate calls. The
            list branch is chosen from ``text`` alone, so the other arguments
            must be lists of matching length.
        """
        ref_audio = ref_speech if ref_speech is not None else speech
        if ref_audio is None:
            raise ValueError(
                "No reference audio provided: set input_key to include "
                "'ref_speech' (cross/same-speaker) or 'speech' (self-reference)."
            )

        if isinstance(text, (list, tuple)):
            ref_text = ref_text if ref_text is not None else [None] * len(text)
            wavs = [
                self.infer_one(t, a, r) for t, a, r in zip(text, ref_audio, ref_text)
            ]
            return {"wav": wavs}
        return {"wav": self.infer_one(text, ref_audio, ref_text)}
