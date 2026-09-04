"""F5-TTS inference engine.

Built by a recipe's ``infer`` stage
(``model._target_: espnet3.systems.tts.f5_tts.inference.F5TTSInference``).
For each test sample the runner calls ``model(**{key: data[key] for key in input_key})``
with ``input_key: [text, reference_speech, reference_text]`` (the cross- and
same-speaker protocol) and feeds the result to ``src.inference.build_output``
(which needs a ``"wav"`` entry).

The model is rebuilt from the *training* config by instantiating that config's
own ``model`` block (``espnet3.systems.tts.f5_tts.f5tts.F5TTS``), so it stays in
sync with whatever was trained. Text is tokenized with the exact espnet2
components used in training (TextCleaner + tokenizer + TokenIDConverter) read
from the training config's preprocessor.
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
from espnet2.torch_utils.safe_torch_load import safe_torch_load
from espnet3.systems.tts.f5_tts import (
    BIGVGAN_DEFAULT_MODEL,
    VOCOS_DEFAULT_MODEL,
)
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _split_on_byte_budget(text: str, max_chars: int) -> List[str]:
    """Cut ``text`` into pieces of at most ``max_chars`` utf-8 bytes.

    Prefers to cut at whitespace so a word is never split across two chunks,
    since each chunk is synthesized as its own utterance and a fragment would
    be spoken as one. Falls back to a character boundary when the budget holds
    no whitespace, which is the normal case for scripts that do not use spaces
    and for a single word wider than the budget. Never cuts inside a character,
    so multi-byte scripts survive intact.
    """
    pieces: List[str] = []
    remaining = text
    while len(remaining.encode("utf-8")) > max_chars:
        # Longest prefix that fits, counted in bytes but cut between characters.
        cut = 0
        size = 0
        for index, character in enumerate(remaining, start=1):
            size += len(character.encode("utf-8"))
            if size > max_chars:
                break
            cut = index
        cut = max(cut, 1)  # always make progress, even on an oversized character
        # Prefer the last word boundary inside that prefix.
        boundary = remaining.rfind(" ", 0, cut)
        if boundary > 0:
            cut = boundary + 1
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        pieces.append(remaining)
    return pieces


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of at most ``max_chars``. Ported from F5-TTS.

    Note:
        The budget is counted in utf-8 bytes, matching upstream F5-TTS
        ``chunk_text``. For ASCII text that equals the character count, but a
        CJK character costs three, so Chinese chunks hold roughly a third of
        ``max_chars`` characters. Unlike upstream, a single sentence longer
        than the budget is split rather than kept whole, so the limit always
        holds.
    """
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
                current_chunk = ""
            if len(sentence.encode("utf-8")) > max_chars:
                # A sentence with no internal punctuation can exceed the budget
                # on its own; upstream keeps it whole, which defeats the point
                # of chunking. Cut it on character boundaries instead.
                *head, sentence = _split_on_byte_budget(sentence, max_chars)
                chunks.extend(piece.strip() for piece in head)
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
        previous_wave = final_wave
        cross_fade_samples = int(cross_fade_duration * sample_rate)
        cross_fade_samples = min(cross_fade_samples, len(previous_wave), len(next_wave))
        if cross_fade_samples <= 0:
            final_wave = np.concatenate([previous_wave, next_wave])
            continue
        previous_overlap = previous_wave[-cross_fade_samples:]
        next_overlap = next_wave[:cross_fade_samples]
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)
        cross = previous_overlap * fade_out + next_overlap * fade_in
        final_wave = np.concatenate(
            [previous_wave[:-cross_fade_samples], cross, next_wave[cross_fade_samples:]]
        )
    return final_wave


class F5TTSInference:
    """Load a trained F5-TTS model + vocoder and synthesize waveforms."""

    def __init__(
        self,
        train_config: str,
        checkpoint_path: str,
        device: str = "cpu",
        use_ema: bool = True,
        vocoder_name: str = "vocos",
        vocoder_path: Optional[str] = None,
        target_sample_rate: int = 24000,
        ode_solver_steps: int = 32,
        guidance_strength: float = 2.0,
        sway_sampling_coefficient: float = -1.0,
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
            checkpoint_path: Lightning checkpoint (``.ckpt``) from training.
            device: Torch device string.
            use_ema: Load EMA-averaged weights (``ema_model_state_dict``) when
                present; otherwise the raw ``state_dict``.
            vocoder_name / vocoder_path: ``"vocos"`` (default) or ``"bigvgan"``.
            target_sample_rate: Output/vocoder sample rate.
            ode_solver_steps: Number of ODE solver steps, upstream's
                ``nfe_step`` (number of function evaluations).
            guidance_strength: Classifier-free guidance scale, upstream's
                ``cfg_strength``.
            sway_sampling_coefficient / speed / seed: Remaining sampling
                hyperparameters forwarded to ``CFM.sample``.
            native_f5: Load an OFFICIAL SWivid/F5-TTS checkpoint (``.pt`` or
                ``.safetensors``) instead of an espnet/Lightning ckpt. The weights
                are loaded straight into the ported CFM (``model.cfm``), so
                the architecture and the pinyin ``token_list`` in
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
                  checkpoint_path: ${exp_dir}/last.ckpt
                  device: cuda
                  ode_solver_steps: 32
                  guidance_strength: 2.0

        Note:
            ``train_config`` is the recipe's own training YAML, not the
            ``config.yaml`` written into ``exp_dir``: the model is rebuilt from
            the ``model:`` block via Hydra, so architecture and tokenizer
            settings always come from one source of truth. Construction is
            eager, loading the checkpoint and the vocoder up front.
        """
        self.device = torch.device(device)
        self.target_sample_rate = target_sample_rate
        self.ode_solver_steps = ode_solver_steps
        self.guidance_strength = guidance_strength
        self.sway_sampling_coefficient = sway_sampling_coefficient
        self.speed = speed
        self.target_rms = target_rms
        self.cross_fade_duration = cross_fade_duration
        self.seed = seed

        config = OmegaConf.to_container(
            load_config_with_defaults(train_config), resolve=True
        )
        feats_extract_config = (config.get("model") or {}).get(
            "feats_extract_config"
        ) or {}
        self.hop_length = int(feats_extract_config.get("hop_length", 256))
        model = self._build_model(config, checkpoint_path, use_ema, native_f5)
        # F5TTS components used for generation.
        self.feats_extract = model.feats_extract
        self.cfm = model.cfm
        self.model = model

        self._build_tokenizer(config)
        self.vocoder = self._load_vocoder(vocoder_name, vocoder_path)

    # ------------------------------------------------------------------ build

    def _build_model(
        self, config: dict, checkpoint_path: str, use_ema: bool, native_f5: bool = False
    ):
        model_config = config.get("model")
        if not model_config or not model_config.get("_target_"):
            raise ValueError(
                "train_config must set `model._target_` (the F5-TTS model)."
            )
        logger.info("Building TTS model via %s", model_config["_target_"])
        model = instantiate(model_config)

        if native_f5:
            # Official SWivid/F5-TTS checkpoint: CFM-level keys (transformer.* /
            # mel_spec.*, EMA prefixed ema_model.). Load straight into the ported
            # CFM so the enclosing model's prefixes (cfm.) don't matter.
            cfm_state_dict = self._load_native_f5_state(checkpoint_path, use_ema)
            missing, unexpected = model.cfm.load_state_dict(
                cfm_state_dict, strict=False
            )
            self._log_model_loading(
                "F5-native -> model.cfm", checkpoint_path, missing, unexpected
            )
            return model.to(self.device).eval()

        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        if use_ema and "ema_model_state_dict" in checkpoint:
            prefix = "ema_model."
            state_dict = {
                key[len(prefix) :]: value
                for key, value in checkpoint["ema_model_state_dict"].items()
                if key.startswith(prefix)
            }
            logger.info("Loading EMA weights from %s", checkpoint_path)
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
            logger.info("Loading raw (non-EMA) weights from %s", checkpoint_path)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self._log_model_loading("espnet", checkpoint_path, missing, unexpected)
        return model.to(self.device).eval()

    @staticmethod
    def _load_native_f5_state(checkpoint_path: str, use_ema: bool) -> dict:
        """CFM-level state dict from an official F5-TTS checkpoint.

        Handles ``.pt`` (``torch.load`` -> ``model_state_dict`` /
        ``ema_model_state_dict``) and ``.safetensors`` (a flat EMA tensor dict).
        Returns keys at CFM level (``transformer.*``, ``mel_spec.*``): the
        ``ema_model.`` prefix is stripped and the ``initted`` / ``step``
        bookkeeping tensors are dropped, mirroring F5's own ``load_checkpoint``.
        """
        if str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            raw = load_file(checkpoint_path)  # flat: ema_model.* (+ initted/step)
        else:
            checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
            if use_ema and "ema_model_state_dict" in checkpoint:
                raw = checkpoint["ema_model_state_dict"]
            elif "model_state_dict" in checkpoint:
                raw = checkpoint["model_state_dict"]
            else:
                raw = checkpoint
        return {
            key.replace("ema_model.", "", 1): value
            for key, value in raw.items()
            if key not in ("initted", "step")
        }

    @staticmethod
    def _log_model_loading(tag: str, checkpoint_path: str, missing, unexpected) -> None:
        logger.info("Loaded %s weights from %s", tag, checkpoint_path)
        if missing:
            logger.warning("[%s] missing keys (%d): %s", tag, len(missing), missing)
        if unexpected:
            logger.warning(
                "[%s] unexpected keys (%d): %s", tag, len(unexpected), unexpected
            )

    def _build_tokenizer(self, config: dict) -> None:
        """Replicate the training-time text tokenization for inference.

        Sets ``self._tokenize(text) -> int64 ids`` matching the training
        preprocessor: F5's zh+en pinyin (``F5PinyinPreprocessor``) or espnet2's
        char/phn tokenizer (``CommonPreprocessor``).
        """
        preprocessor_config = config.get("dataset", {}).get("preprocessor", {})
        target = str(preprocessor_config.get("_target_", ""))

        if "F5PinyinPreprocessor" in target or preprocessor_config.get("vocab_file"):
            # F5 zh+en pinyin: F5's own tokenizer + vocab (unknown token -> 0).
            from espnet3.systems.tts.f5_tts.pinyin import (
                load_vocab_char_map,
                text_to_pinyin_ids,
            )

            vocab_char_map = load_vocab_char_map(preprocessor_config["vocab_file"])
            self._tokenize = lambda text: text_to_pinyin_ids(text, vocab_char_map)
            return

        # espnet2 char/phn tokenization (matches CommonPreprocessor).
        token_list = preprocessor_config.get("token_list")
        if token_list is None:
            raise ValueError(
                "Could not find dataset.preprocessor.token_list in train_config."
            )
        if preprocessor_config.get("g2p_type") == "f5_pinyin":
            from espnet3.systems.tts.f5_tts.pinyin import register_f5_pinyin_g2p

            register_f5_pinyin_g2p()
        cleaner = TextCleaner(preprocessor_config.get("text_cleaner"))
        tokenizer = build_tokenizer(
            token_type=preprocessor_config.get("token_type", "char"),
            bpemodel=preprocessor_config.get("bpemodel"),
            non_linguistic_symbols=preprocessor_config.get("non_linguistic_symbols"),
            g2p_type=preprocessor_config.get("g2p_type"),
        )
        token_id_converter = TokenIDConverter(token_list)
        self._tokenize = lambda text: np.asarray(
            token_id_converter.tokens2ids(tokenizer.text2tokens(cleaner(text))),
            dtype=np.int64,
        )

    def _load_vocoder(self, vocoder_name: str, vocoder_path: Optional[str]):
        if vocoder_name == "vocos":
            try:
                from vocos import Vocos
            except ImportError as error:
                raise ImportError(
                    "vocos is required for vocoder_name='vocos'. Install with "
                    "`pip install vocos`."
                ) from error
            if vocoder_path:
                vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
                state = safe_torch_load(
                    f"{vocoder_path}/pytorch_model.bin", map_location="cpu"
                )
                vocoder.load_state_dict(state)
            else:
                vocoder = Vocos.from_pretrained(VOCOS_DEFAULT_MODEL)
        elif vocoder_name == "bigvgan":
            try:
                import bigvgan
            except ImportError as error:
                raise ImportError(
                    "bigvgan is required for vocoder_name='bigvgan'. See "
                    "https://github.com/NVIDIA/BigVGAN."
                ) from error
            repo = vocoder_path or BIGVGAN_DEFAULT_MODEL
            vocoder = bigvgan.BigVGAN.from_pretrained(repo, use_cuda_kernel=False)
            vocoder.remove_weight_norm()
        else:
            raise ValueError(f"Unsupported vocoder: {vocoder_name!r}.")
        return vocoder.to(self.device).eval()

    # -------------------------------------------------------------- inference

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
        target_text: str,
        reference_audio: np.ndarray,
        reference_text: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesize ``target_text`` in the voice of ``reference_audio``.

        Mirrors upstream F5-TTS ``infer_process``: RMS-normalize the reference,
        split the target text into reference-length-dependent chunks, sample
        each chunk, vocode, then cross-fade the pieces back together.

        Args:
            target_text: Target text to speak.
            reference_audio: Reference waveform at ``target_sample_rate``. Multi-
                channel input is averaged down to mono.
            reference_text: Transcript of ``reference_audio``. Defaults to
                ``target_text``, treating the reference as self-referential.

        Returns:
            Mono waveform as ``float32`` at ``target_sample_rate``. Returns a
            single zero sample when the text yields no synthesizable chunk.

        Example:
            .. code-block:: python

                >>> wav = tts.infer_one(
                ...     "hello world", reference_audio, reference_text="hi"
                ... )
                >>> wav.dtype, wav.ndim
                (dtype('float32'), 1)

        Note:
            Output length is governed by the reference: the per-chunk duration
            is extrapolated from the reference audio-to-text ratio, so a
            mismatched ``reference_text`` skews it. The reference loudness is restored
            after vocoding, so the result matches the input level rather than
            ``target_rms``.
        """
        reference_text = target_text if reference_text is None else reference_text
        sample_rate = self.target_sample_rate

        # Reference waveform [1, T]: mono + RMS normalization to target_rms.
        audio = torch.as_tensor(np.asarray(reference_audio), dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        # A silent reference has rms == 0; scaling by target_rms / 0 would make
        # the whole waveform NaN and propagate silently through to the output.
        normalized = 0 < rms < self.target_rms
        if normalized:
            audio = audio * self.target_rms / rms
        audio = audio.to(self.device)

        # F5 appends a trailing space when reference_text ends on a single-byte char.
        if reference_text and len(reference_text[-1].encode("utf-8")) == 1:
            reference_text = reference_text + " "

        # Reference-length-dependent chunking (same formula as infer_process).
        audio_seconds = audio.shape[-1] / sample_rate
        max_target_chars = int(
            len(reference_text.encode("utf-8"))
            / audio_seconds
            * (22 - audio_seconds)
            * self.speed
        )
        target_text_chunks = _chunk_text(
            target_text, max_chars=max(max_target_chars, 1)
        )
        if not target_text_chunks:
            return np.zeros(1, dtype=np.float32)

        # Measure the prompt with the mel CFM will use, not samples // hop:
        # the vocos front end is centre-padded and yields one frame more, so
        # the sample-count formula leaves a prompt frame in the output.
        reference_mel_length = self.cfm.mel_spec(audio).shape[-1]
        reference_text_bytes = max(len(reference_text.encode("utf-8")), 1)

        generated_waves: List[np.ndarray] = []
        for target_chunk in target_text_chunks:
            # F5 slows down very short generations.
            local_speed = 0.3 if len(target_chunk.encode("utf-8")) < 10 else self.speed
            target_chunk_bytes = len(target_chunk.encode("utf-8"))
            duration = reference_mel_length + int(
                reference_mel_length
                / reference_text_bytes
                * target_chunk_bytes
                / local_speed
            )

            token_ids = (
                torch.from_numpy(self._tokenize(reference_text + target_chunk))
                .unsqueeze(0)
                .to(self.device)
            )
            # Pass the raw reference wave as cond; CFM.sample extracts its mel
            # internally. F5TTS defaults CFM's MelSpec from feats_extract, so
            # the two agree unless the config deliberately diverges.
            out, _ = self.cfm.sample(
                cond=audio,
                text=token_ids,
                duration=duration,
                steps=self.ode_solver_steps,
                cfg_strength=self.guidance_strength,
                sway_sampling_coef=self.sway_sampling_coefficient,
                seed=self.seed,
            )

            generated_mel = out[:, reference_mel_length:, :].to(
                torch.float32
            )  # drop prompt
            if generated_mel.shape[1] == 0:
                continue
            wave = self._vocode(generated_mel.permute(0, 2, 1))  # [1, d, n_gen]
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            if normalized:  # de-normalize back to the reference loudness
                wave = wave * float(rms / self.target_rms)
            generated_waves.append(wave)

        if not generated_waves:
            return np.zeros(1, dtype=np.float32)
        return _cross_fade(generated_waves, self.cross_fade_duration, sample_rate)

    def __call__(
        self,
        text: Union[str, List[str]],
        reference_speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        reference_text: Optional[Union[str, List[str]]] = None,
        speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> dict:
        """Inference entry point used by the runner.

        ``text`` is the target text. The reference audio comes from ``reference_speech``
        (cross/same-speaker protocol, with ``reference_text`` its transcript); if only
        ``speech`` is given it is used as the reference and ``reference_text`` defaults
        to ``text`` (self-reference). Supports a single sample (``batch_size:
        null``) or a list (batched).

        Args:
            text: Target text, or a list of them for a batched call.
            reference_speech: Reference waveform(s) for the cross/same-speaker
                protocol.
            reference_text: Transcript(s) of ``reference_speech``.
            speech: Fallback reference used when ``reference_speech`` is absent, which
                makes the call self-referential.

        Returns:
            ``{"wav": waveform}`` for a single sample, or ``{"wav": [...]}``
            with one waveform per input when ``text`` is a list.

        Raises:
            ValueError: If neither ``reference_speech`` nor ``speech`` is given
                (F5 is zero-shot and cannot synthesize without a reference), or
                if the batched inputs have mismatched lengths.

        Example:
            .. code-block:: python

                >>> tts(text="hello", speech=ref_wave)["wav"].ndim
                1
                >>> len(tts(text=["a", "b"], reference_speech=[w1, w2])["wav"])
                2

        Note:
            Batched input is looped over ``infer_one`` rather than batched
            through the solver, so it costs the same as separate calls. The
            list branch is chosen from ``text`` alone; the other arguments
            must be lists of matching length or a ``ValueError`` is raised.
        """
        reference_audio = reference_speech if reference_speech is not None else speech
        if reference_audio is None:
            raise ValueError(
                "No reference audio provided: set input_key to include "
                "'reference_speech' (cross/same-speaker) or 'speech' (self-reference)."
            )

        if isinstance(text, (list, tuple)):
            reference_text = (
                reference_text if reference_text is not None else [None] * len(text)
            )
            # zip() would truncate to the shortest input, silently returning
            # fewer waveforms than requested and misaligning them with the
            # runner's test samples.
            if len(reference_audio) != len(text) or len(reference_text) != len(text):
                raise ValueError(
                    "Batched inputs must have matching lengths: got "
                    f"{len(text)} text, {len(reference_audio)} reference audio, "
                    f"{len(reference_text)} reference text."
                )
            wavs = [
                self.infer_one(one_text, one_reference_audio, one_reference_text)
                for one_text, one_reference_audio, one_reference_text in zip(
                    text, reference_audio, reference_text
                )
            ]
            return {"wav": wavs}
        return {"wav": self.infer_one(text, reference_audio, reference_text)}
