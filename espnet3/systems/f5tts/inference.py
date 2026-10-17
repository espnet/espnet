"""F5-TTS inference: the engine and the model a packed bundle names.

:class:`F5TTSInference` is the engine: it loads a trained model and the
vocoder and synthesizes waveforms from a mono float32 reference at its own
rate. :class:`Inference` wraps it for the ``infer`` stage, for
:class:`espnet3.publication.InferenceModel` and for the demo: it is built from
the engine's own arguments by Hydra, accepts the reference the way a caller
holds it (a path, the ``(rate, samples)`` pair ``gr.Audio`` returns, an array
or a tensor) and resamples it to the engine's rate, and refuses a missing
input by name.

A recipe's ``infer`` stage names :class:`Inference` as its ``model`` and
gives ``input_key`` (``text``, ``reference_speech``, ``reference_text``), an
``output_fn`` that stamps the sample's ``utt_id`` onto the result
(``egs3.TEMPLATE.f5tts.src.inference.build_output``) and ``output_artifacts``
with ``wav: {type: wav, sample_rate: 24000}`` so ``wav.scp`` points at audio
files. The bare engine can still be named instead; it then takes the
reference as a bare array at its own rate.

The model is rebuilt from the *training* config by instantiating that config's
own ``model`` block (``espnet3.systems.f5tts.src.f5tts.F5TTS``), so it stays in
sync with whatever was trained. Text is tokenized with the exact espnet2
components used in training (TextCleaner + tokenizer + TokenIDConverter) read
from the training config's preprocessor.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.safe_torch_load import safe_torch_load
from espnet3.contract import Field
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.f5tts import VOCOS_DEFAULT_MODEL
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
        vocoder_path: Optional[str] = None,
        target_sample_rate: Optional[int] = None,
        ode_solver_steps: int = 32,
        guidance_strength: float = 2.0,
        sway_sampling_coefficient: float = -1.0,
        speed: float = 1.0,
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
        seed: Optional[int] = None,
    ):
        """Build the model, tokenizer and vocoder for inference.

        Args:
            train_config: Path to the training config the checkpoint was
                trained with; its ``model`` block and its preprocessor
                tokenization settings rebuild the model and the tokenizer.
            checkpoint_path: Lightning checkpoint (``.ckpt``) from training.
            device: Torch device string.
            use_ema: Load EMA-averaged weights (``ema_model_state_dict``) when
                present; otherwise the raw ``state_dict``.
            vocoder_path: Local directory holding the Vocos ``config.yaml``
                and ``pytorch_model.bin``. When unset the default checkpoint is
                fetched from the Hugging Face Hub.
            target_sample_rate: Rate of the reference audio and of the
                synthesized waveform. It is fixed by the model's mel front end
                (``feats_extract_config.fs``), so leave it unset to use that
                rate; a value that differs is refused.
            ode_solver_steps: Number of ODE solver steps, upstream's
                ``nfe_step`` (number of function evaluations).
            guidance_strength: Classifier-free guidance scale, upstream's
                ``cfg_strength``.
            sway_sampling_coefficient: Sway sampling coefficient of the ODE
                time steps, upstream's ``sway_sampling_coef``; forwarded to
                ``CFM.sample``.
            speed: Speaking-rate factor: the output duration estimated from
                the reference is divided by it, so above 1 speaks faster, and
                it scales the text length of each chunk. A chunk shorter than
                10 bytes always uses 0.3.
            target_rms: Loudness a quieter reference is raised to before
                synthesis; the output is scaled back to the reference's own
                level afterwards.
            cross_fade_duration: Seconds of linear cross-fade between the
                waveforms of consecutive text chunks; ``0`` concatenates them.
            seed: Random seed of the initial noise, for reproducible output;
                ``None`` leaves it unseeded.

        Raises:
            ValueError: If ``train_config`` has no ``model._target_`` or if
                its ``dataset.preprocessor.token_list`` is missing; if
                ``target_sample_rate`` differs from the model's rate; or if
                the vocoder was trained at a rate other than the model's.

        Example:
            .. code-block:: yaml

                model:
                  _target_: espnet3.systems.f5tts.inference.F5TTSInference
                  train_config: ${exp_dir}/config.yaml
                  checkpoint_path: ${exp_dir}/last.ckpt
                  device: cuda
                  ode_solver_steps: 32
                  guidance_strength: 2.0

        Note:
            A recipe passes ``${exp_dir}/config.yaml``, the training config
            ``F5TTSSystem.train`` writes beside the checkpoint, so the model is
            rebuilt from the exact config it was trained with; its ``model:``
            block is instantiated via Hydra. A recipe normally names
            :class:`Inference`, which wraps this engine, rather than the
            engine itself. Construction is eager, loading the checkpoint and
            the vocoder up front.
        """
        self.device = torch.device(device)
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
        model = self._build_model(config, checkpoint_path, use_ema)
        # F5TTS components used for generation.
        self.feats_extract = model.feats_extract
        self.cfm = model.cfm
        self.model = model

        self._build_tokenizer(config)
        self.vocoder = self._load_vocoder(vocoder_path)
        self.target_sample_rate = self._check_sample_rate(target_sample_rate)

    # ------------------------------------------------------------------ build

    def _build_model(self, config: dict, checkpoint_path: str, use_ema: bool):
        """Instantiate ``config.model`` and load the checkpoint, EMA weights first.

        Falls back to the raw ``state_dict`` when the checkpoint has no EMA
        state or ``use_ema`` is false. Loads with ``strict=False`` and logs
        the missing/unexpected keys.
        """
        model_config = config.get("model")
        if not model_config or not model_config.get("_target_"):
            raise ValueError(
                "train_config must set `model._target_` (the F5-TTS model)."
            )
        logger.info("Building TTS model via %s", model_config["_target_"])
        model = instantiate(model_config)

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
    def _log_model_loading(tag: str, checkpoint_path: str, missing, unexpected) -> None:
        """Log a load summary, warning on missing or unexpected keys."""
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
            from espnet3.systems.f5tts.src.pinyin import (
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
            from espnet3.systems.f5tts.src.pinyin import register_f5_pinyin_g2p

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

    def _check_sample_rate(self, requested: Optional[int]) -> int:
        """Return the model's sample rate, refusing a rate that contradicts it.

        The mel front end fixes the rate the model reads the reference at, and
        the vocoder the rate it writes the waveform at; a different requested
        rate would resample the reference and label the output wrongly.
        """
        model_rate = int(self.feats_extract.fs)
        vocoder_rate = _vocoder_sample_rate(self.vocoder)
        if vocoder_rate is not None and vocoder_rate != model_rate:
            raise ValueError(
                f"The vocoder produces {vocoder_rate} Hz audio, but the model's "
                f"mel front end is configured for {model_rate} Hz."
            )
        if requested is not None and int(requested) != model_rate:
            raise ValueError(
                f"target_sample_rate={requested} does not match the model's "
                f"{model_rate} Hz (feats_extract_config.fs); leave it unset."
            )
        return model_rate

    def _load_vocoder(self, vocoder_path: Optional[str]):
        """Load Vocos from ``vocoder_path`` or the default pretrained checkpoint."""
        from vocos import Vocos

        if vocoder_path:
            vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
            state = safe_torch_load(
                f"{vocoder_path}/pytorch_model.bin", map_location="cpu"
            )
            vocoder.load_state_dict(state)
        else:
            vocoder = Vocos.from_pretrained(VOCOS_DEFAULT_MODEL)
        return vocoder.to(self.device).eval()

    # -------------------------------------------------------------- inference

    def _vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocode mel ``[1, d, n]`` to waveform ``[nw]``."""
        wav = self.vocoder.decode(mel)
        return wav.squeeze().detach().cpu()

    @torch.no_grad()
    def infer_one(
        self,
        target_text: str,
        reference_audio: np.ndarray,
        reference_text: str,
    ) -> np.ndarray:
        """Synthesize ``target_text`` in the voice of ``reference_audio``.

        Mirrors upstream F5-TTS ``infer_process``: RMS-normalize the reference,
        split the target text into reference-length-dependent chunks, sample
        each chunk, vocode, then cross-fade the pieces back together.

        Args:
            target_text: Target text to speak.
            reference_audio: Reference waveform at ``target_sample_rate``. Multi-
                channel input is averaged down to mono.
            reference_text: Transcript of ``reference_audio``. Required: the
                model reads the reference through it, and the output duration
                is extrapolated from the reference's audio-to-text ratio.

        Returns:
            Mono waveform as ``float32`` at ``target_sample_rate``. Returns a
            single zero sample when the text yields no synthesizable chunk.

        Raises:
            ValueError: If ``reference_text`` is empty.

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
        if not reference_text or not reference_text.strip():
            raise ValueError(
                "reference_text is required: the transcript of reference_audio. "
                "A wrong or missing transcript misaligns the prompt and skews "
                "the output duration."
            )
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
        """Inference entry point for the bare engine.

        ``text`` is the target text; the reference audio comes from
        ``reference_speech`` (or ``speech``, an older name for the same
        field), and ``reference_text`` is its transcript, which is required.
        Supports a single sample (``batch_size: null``) or a list (batched).

        Args:
            text: Target text, or a list of them for a batched call.
            reference_speech: Reference waveform(s), the voice to clone.
            reference_text: Transcript(s) of the reference. Required.
            speech: The reference waveform(s) under the dataset column name
                ``speech``, used when ``reference_speech`` is absent.

        Returns:
            ``{"wav": waveform}`` for a single sample, or ``{"wav": [...]}``
            with one waveform per input when ``text`` is a list.

        Raises:
            ValueError: If neither ``reference_speech`` nor ``speech`` is given
                (F5 is zero-shot and cannot synthesize without a reference), if
                ``reference_text`` is missing, or if the batched inputs have
                mismatched lengths.

        Example:
            .. code-block:: python

                >>> tts(text="hello", speech=ref_wave, reference_text="hi")["wav"].ndim
                1
                >>> wavs = tts(
                ...     text=["a", "b"],
                ...     reference_speech=[w1, w2],
                ...     reference_text=["x", "y"],
                ... )["wav"]
                >>> len(wavs)
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
                "No reference audio provided: pass 'reference_speech' (or "
                "'speech', an older name for the same field)."
            )

        if reference_text is None:
            raise ValueError(
                "No reference transcript provided: pass 'reference_text', the "
                "transcript of the reference audio."
            )

        if isinstance(text, (list, tuple)):
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


def _vocoder_sample_rate(vocoder: Any) -> Optional[int]:
    """Return the rate a Vocos vocoder was trained at, or ``None`` if unknown.

    Vocos exposes it on its mel feature extractor; a vocoder without that
    attribute is not checked.
    """
    feature_extractor = getattr(vocoder, "feature_extractor", None)
    rate = getattr(getattr(feature_extractor, "mel_spec", None), "sample_rate", None)
    return int(rate) if rate is not None else None


class Inference:
    """Synthesize with an F5-TTS model: the engine plus input handling.

    The model behind an ``f5tts`` bundle is :class:`F5TTSInference`, which
    this class builds and calls. On top of the engine it does what a caller
    should not have to think about: reading a reference file, accepting the
    ``(rate, samples)`` pair Gradio hands over, resampling the reference to
    the engine's rate and taking one channel, and refusing a missing input
    by name instead of synthesizing from nothing.

    Built two ways, both ending in :attr:`backend`:

    - ``Inference(train_config=..., checkpoint_path=..., ...)`` from
      :class:`F5TTSInference`'s own arguments, which is what
      ``inference.yaml`` does and what Hydra calls (``device`` included);
    - ``Inference(f5tts_inference)`` around an engine already built.

    Called with the target text, a reference utterance and its transcript it
    returns ``{"wav": samples}``: a mono ``float32`` waveform at
    :attr:`sample_rate`. Lists of the three inputs run item by item and
    return one dict per item, which is the batched shape the runner writes.

    Args:
        backend: An engine already built; exclusive with the keyword
            arguments.
        **engine_kwargs: :class:`F5TTSInference`'s arguments
            (``train_config``, ``checkpoint_path``, ``device``,
            ``ode_solver_steps``, ...).

    Raises:
        TypeError: If both ``backend`` and keyword arguments are given, or
            neither.

    Example:
        .. code-block:: python

            >>> from espnet3.publication import InferenceModel
            >>> model = InferenceModel.from_packed("exp/training/model_pack")
            >>> output = model(
            ...     {"text": "hello world", "reference_speech": "prompt.wav",
            ...      "reference_text": "the prompt transcript"}
            ... )
            >>> output["wav"].dtype, output["wav"].ndim
            (dtype('float32'), 1)
            >>> model.model.sample_rate
            24000

        In ``inference.yaml``, for the ``infer`` stage:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.f5tts.inference.Inference
              train_config: ${exp_dir}/config.yaml  # written by F5TTSSystem.train
              checkpoint_path: ${exp_dir}/last.ckpt
              ode_solver_steps: 32
            input_key: [text, reference_speech, reference_text]
            output_fn: egs3.TEMPLATE.f5tts.src.inference.build_output
            output_artifacts:
              wav: {type: wav, sample_rate: 24000}

    Note:
        All three inputs are required. F5-TTS reads the reference through its
        transcript and sets the output duration from the reference's
        audio-to-text ratio, so a missing transcript is refused rather than
        guessed. A bare array or tensor reference is taken to be at
        :attr:`sample_rate` already, since it carries no rate of its own; a
        path or a ``(rate, samples)`` pair is resampled.
    """

    #: What a call takes, in the order ``inference.yaml`` lists ``input_key``
    #: and ``demo.yaml`` lists ``ui.inputs``.
    inputs = (
        Field("text", "text", "Text to synthesize"),
        Field("reference_speech", "audio", "Reference speech"),
        Field("reference_text", "text", "Reference transcript"),
    )
    #: What a call returns: the synthesized waveform.
    outputs = (Field("wav", "audio", "Synthesized speech"),)

    def __init__(self, backend: Optional[F5TTSInference] = None, **engine_kwargs):
        """Wrap an engine, or build one from :class:`F5TTSInference`'s arguments."""
        if backend is not None and engine_kwargs:
            raise TypeError(
                "Inference takes either a built engine or the engine's own "
                "arguments, not both"
            )
        if backend is None:
            if not engine_kwargs:
                raise TypeError(
                    "Inference needs an engine or F5TTSInference's arguments "
                    "(train_config, checkpoint_path, ...)"
                )
            backend = F5TTSInference(**engine_kwargs)
        self.backend = backend

    @property
    def sample_rate(self) -> int:
        """Return the rate of the reference the engine takes and the audio it gives.

        Returns:
            The engine's ``target_sample_rate``, the vocoder's rate.
        """
        return int(self.backend.target_sample_rate)

    @staticmethod
    def _require(name: str, value: Any) -> Any:
        """Return ``value``, or raise ``TypeError`` naming the missing input."""
        if value is None:
            raise TypeError(f"{name!r} is required and was not given")
        return value

    def _reference(self, value: Any) -> np.ndarray:
        """Return the reference as a mono float32 array at :attr:`sample_rate`.

        A path is read, a ``(rate, samples)`` pair or an :class:`Audio` is
        resampled, a bare array or tensor is taken as already at the rate.
        Of a multichannel reference the first channel is kept.
        """
        try:
            audio = Audio.coerce(value, self.sample_rate)
        except TypeError as error:
            raise TypeError(f"'reference_speech': {error}") from error
        return audio.mono().array

    def _one(self, text: Any, reference_speech: Any, reference_text: Any) -> dict:
        """Synthesize one item after checking and converting its inputs."""
        text = self._require("text", text)
        reference_speech = self._require("reference_speech", reference_speech)
        reference_text = self._require("reference_text", reference_text)
        if not isinstance(text, str) or not isinstance(reference_text, str):
            raise TypeError("'text' and 'reference_text' must be str")
        reference = self._reference(reference_speech)
        return {"wav": self.backend.infer_one(text, reference, reference_text)}

    def __call__(
        self,
        text: Any = None,
        reference_speech: Any = None,
        reference_text: Any = None,
    ) -> Union[dict, List[dict]]:
        """Synthesize ``text`` in the voice of ``reference_speech``.

        Args:
            text: The target text, or a list of them for a batch.
            reference_speech: The voice to clone: a path, a ``(rate,
                samples)`` pair, an array or tensor at :attr:`sample_rate`,
                or an :class:`~espnet3.contract.kinds.audio.Audio`; a list
                of those for a batch.
            reference_text: Transcript of ``reference_speech``, or a list.

        Returns:
            ``{"wav": samples}`` for one item: a mono ``float32`` waveform at
            :attr:`sample_rate`. For lists, one such dict per item, in order.

        Raises:
            TypeError: If an input is missing (``None``), not of its kind, or
                the batched lists differ in length.

        Example:
            .. code-block:: python

                >>> output = model("hello world", "prompt.wav", "a transcript")
                >>> output["wav"].dtype, output["wav"].ndim
                (dtype('float32'), 1)
        """
        if isinstance(text, list):
            if not (
                isinstance(reference_speech, list) and isinstance(reference_text, list)
            ):
                raise TypeError("a batched call needs lists for all three inputs")
            if not len(text) == len(reference_speech) == len(reference_text):
                raise TypeError(
                    "batched inputs differ in length: "
                    f"{len(text)} text, {len(reference_speech)} reference_speech, "
                    f"{len(reference_text)} reference_text"
                )
            return [
                self._one(t, s, r)
                for t, s, r in zip(text, reference_speech, reference_text)
            ]
        return self._one(text, reference_speech, reference_text)
