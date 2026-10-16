"""End-to-end kNN-VC inference model (WavLM encoder -> kNN -> HiFi-GAN)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.systems.knnvc.checkpoint import load_generator_state_dict
from espnet3.systems.knnvc.matcher import match_features
from espnet3.systems.knnvc.vocoder import (
    KNNVCGenerator,
    build_generator_params,
)
from espnet3.systems.knnvc.wavlm_encoder import WavLMEncoder

logger = logging.getLogger(__name__)

# torchaudio's ITU-R BS.1770 loudness needs at least one 400 ms gating block.
_MIN_LOUDNESS_SECONDS = 0.4


class KNNVCModel:
    """Any-to-any voice conversion with kNN-VC, callable per utterance.

    This is the ``model:`` target of the ``infer`` stage. ESPnet3's
    ``InferenceRunner`` calls it as ``model(**inputs)`` with the dataset fields
    named by ``inference_config.input_key``. A recipe's ``conversion`` dataset
    must provide ``speech`` (the source utterance), ``reference_speech`` (the
    target speaker's utterances) and, optionally, ``target_speaker`` as a cache
    key.

    Args:
        vocoder_checkpoint: Path or URL of the HiFi-GAN generator weights. Any
            layout accepted by
            :func:`espnet3.systems.knnvc.checkpoint.load_generator_state_dict`
            works, so both an ESPnet3 training checkpoint and the official
            ``prematch_g_02500000.pt`` can be used.
        encoder: Optional frozen feature encoder, either a built instance or a
            Hydra config. Leave it unset for the paper's setup, which builds a
            :class:`espnet3.systems.knnvc.wavlm_encoder.WavLMEncoder`
            from ``wavlm_checkpoint``/``layer``; set it to run kNN-VC on
            another self-supervised encoder, which must expose
            ``encode(speech, pad_to_hop)``, ``sample_rate`` and ``device``. The
            features the vocoder was trained on and the ones produced here must
            come from the same encoder.
        wavlm_checkpoint: Path or URL of the WavLM checkpoint, required
            unless ``encoder`` is given. Recipes set it through the
            ``wavlm_checkpoint`` key; see ``egs3/TEMPLATE/knnvc/conf``.
        layer: 1-based WavLM layer used for matching and synthesis (``6``).
            Ignored when ``encoder`` is given.
        generator: Overrides for the HiFi-GAN generator hyperparameters; must
            match the ones the checkpoint was trained with.
        topk: ``k`` of the kNN regression.
        vad_trigger_level: ``torchaudio.transforms.Vad`` trigger level used to
            trim leading and trailing silence from every *reference* utterance
            before encoding (the official default is ``7``). Values ``<= 0``
            disable trimming. The source utterance is never trimmed.
        tgt_loudness_db: Loudness (LUFS) the converted waveform is normalized
            to; ``None`` disables normalization.
        max_cached_speakers: How many target speakers' matching sets to keep
            on ``device``. Encoding a 5-minute reference set costs roughly
            60 MB of GPU memory, so the cache is bounded; the default of ``1``
            is enough because the recipe's conversion pairs are ordered by
            target speaker. Set ``0`` to disable caching.
        device: Device for both the encoder and the vocoder. The ``infer``
            stage passes this automatically.

    Examples:
        Inference config fragment:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.knnvc.model.KNNVCModel
              vocoder_checkpoint: ${exp_dir}/valid.mel_loss.ave_3best.pth
              wavlm_checkpoint: /path/to/WavLM-Large.pt
              topk: 4
            input_key: [speech, reference_speech, target_speaker]

        Direct use:

        >>> model = KNNVCModel("prematch_g_02500000.pt", device="cuda")
        >>> wav = model(source_wav, [ref_wav_1, ref_wav_2])
    """

    def __init__(
        self,
        vocoder_checkpoint: str | Path,
        encoder: Optional[Dict[str, Any]] = None,
        wavlm_checkpoint: Optional[str | Path] = None,
        layer: int = 6,
        generator: Optional[Dict[str, Any]] = None,
        topk: int = 4,
        vad_trigger_level: float = 7.0,
        tgt_loudness_db: Optional[float] = -16.0,
        max_cached_speakers: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        """Load the encoder and the vocoder generator onto ``device``."""
        self.device = torch.device(device)
        self.topk = int(topk)
        self.vad_trigger_level = float(vad_trigger_level)
        self.tgt_loudness_db = tgt_loudness_db
        self.max_cached_speakers = int(max_cached_speakers)

        self.encoder = self.build_encoder(
            encoder=encoder,
            wavlm_checkpoint=wavlm_checkpoint,
            layer=layer,
            device=self.device,
        )
        self.sample_rate = int(self.encoder.sample_rate)

        self.generator = KNNVCGenerator(**build_generator_params(generator))
        self.generator.load_state_dict(load_generator_state_dict(vocoder_checkpoint))
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.generator.to(self.device)
        logger.info(
            "HiFi-GAN generator loaded from %s (%d params)",
            vocoder_checkpoint,
            sum(p.numel() for p in self.generator.parameters()),
        )
        self._matching_set_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    @staticmethod
    def build_encoder(
        encoder: Any,
        wavlm_checkpoint: str | Path,
        layer: int,
        device: torch.device,
    ):
        """Build the frozen encoder: a given instance, a config, or WavLM.

        Args:
            encoder: An encoder instance, a Hydra config with ``_target_``, or
                ``None`` to build WavLM from the remaining arguments. An
                instance is accepted because Hydra builds a nested ``_target_``
                block before the caller's constructor runs, so an ``encoder:``
                block in the inference config arrives here already built.
            wavlm_checkpoint: WavLM checkpoint used when ``encoder`` is
                ``None``; required in that case.
            layer: WavLM layer used when ``encoder`` is ``None``.
            device: Device an encoder built here is placed on. An encoder
                passed in as an instance is used as-is, on whatever device it
                already lives.

        Returns:
            The encoder instance.

        Raises:
            TypeError: If ``encoder`` is neither ``None``, a mapping with
                ``_target_``, nor an object exposing ``encode``.
        """
        if encoder is None:
            if wavlm_checkpoint is None:
                raise ValueError(
                    "Either `encoder` or `wavlm_checkpoint` must be given. "
                    "Recipes set `wavlm_checkpoint` in their config; see "
                    "egs3/TEMPLATE/knnvc/conf."
                )
            return WavLMEncoder(checkpoint=wavlm_checkpoint, layer=layer, device=device)
        if OmegaConf.is_config(encoder):
            encoder = OmegaConf.to_container(encoder, resolve=True)
        if isinstance(encoder, Mapping):
            return instantiate(dict(encoder), device=device)
        if callable(getattr(encoder, "encode", None)):
            # Hydra already instantiated the nested `encoder:` block.
            return encoder
        raise TypeError(
            "`encoder` must be None, a Hydra config with `_target_`, or an "
            f"object with an `encode` method; got {type(encoder).__name__}."
        )

    def get_features(self, speech: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode one waveform into ``(frames, dim)`` encoder features."""
        return self.encoder.encode(speech, pad_to_hop=False)

    def trim_silence(self, speech: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Trim leading and trailing silence with torchaudio's VAD.

        Mirrors ``KNeighborsVC.get_features(..., vad_trigger_level)`` of the
        official implementation: the VAD is applied to the waveform and then to
        its time-reversal, so both ends are trimmed. A trigger level ``<= 0``
        returns the input unchanged (as a float32 tensor).

        Args:
            speech: Mono waveform, 1-D or ``(1, samples)``.

        Returns:
            Trimmed 1-D float32 waveform on CPU. If the VAD removes everything,
            the untrimmed waveform is returned so the reference set never
            loses an utterance.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(np.ascontiguousarray(speech))
        speech = speech.detach().cpu().to(torch.float32).flatten()
        if self.vad_trigger_level <= 1e-3 or speech.numel() == 0:
            return speech
        vad = torchaudio.transforms.Vad(
            sample_rate=self.sample_rate, trigger_level=self.vad_trigger_level
        )
        front_trimmed = vad(speech[None]).flatten()
        if front_trimmed.numel() == 0:
            return speech
        end_trimmed = torch.flip(vad(torch.flip(front_trimmed, (-1,))[None]), (-1,))
        trimmed = end_trimmed.flatten()
        return trimmed if trimmed.numel() > 0 else speech

    def get_matching_set(
        self,
        reference_speech: Sequence[np.ndarray | torch.Tensor],
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode and concatenate the reference utterances of one speaker.

        Each reference is silence-trimmed with :meth:`trim_silence` first
        (``vad_trigger_level``), as in the official implementation.

        Args:
            reference_speech: One or more 16 kHz waveforms of the target
                speaker. A single array is accepted as well.
            cache_key: When given (for example the target speaker id), the
                resulting matching set is cached so that converting several
                sources to the same speaker encodes the references only once.
                At most :attr:`max_cached_speakers` sets are kept, least
                recently used evicted first.

        Returns:
            Matching set of shape ``(total_frames, dim)`` on ``self.device``.

        Raises:
            ValueError: If no reference utterance is given.
        """
        if cache_key is not None and cache_key in self._matching_set_cache:
            self._matching_set_cache.move_to_end(cache_key)
            return self._matching_set_cache[cache_key]
        if isinstance(reference_speech, (np.ndarray, torch.Tensor)):
            reference_speech = [reference_speech]
        if len(reference_speech) == 0:
            raise ValueError("reference_speech must contain at least one waveform.")
        matching_set = torch.cat(
            [self.get_features(self.trim_silence(wav)) for wav in reference_speech],
            dim=0,
        )
        if cache_key is not None and self.max_cached_speakers > 0:
            self._matching_set_cache[cache_key] = matching_set
            while len(self._matching_set_cache) > self.max_cached_speakers:
                self._matching_set_cache.popitem(last=False)
        return matching_set

    @torch.inference_mode()
    def vocode(self, feats: torch.Tensor) -> torch.Tensor:
        """Vocode ``(frames, dim)`` features into a ``(samples,)`` waveform."""
        return self.generator.inference(feats.to(self.device))

    def _normalize_loudness(self, wav: torch.Tensor) -> torch.Tensor:
        if self.tgt_loudness_db is None:
            return wav
        if wav.numel() < int(_MIN_LOUDNESS_SECONDS * self.sample_rate):
            return wav
        loudness = torchaudio.functional.loudness(wav[None], self.sample_rate)
        if not torch.isfinite(loudness):
            return wav
        return torchaudio.functional.gain(wav, float(self.tgt_loudness_db) - loudness)

    @torch.inference_mode()
    def __call__(
        self,
        speech: np.ndarray | torch.Tensor,
        reference_speech: Sequence[np.ndarray | torch.Tensor],
        target_speaker: Optional[str] = None,
    ) -> np.ndarray:
        """Convert ``speech`` to the speaker of ``reference_speech``.

        Args:
            speech: Source waveform, 16 kHz mono.
            reference_speech: Target speaker waveforms (see
                :meth:`get_matching_set`).
            target_speaker: Optional cache key for the reference features.

        Returns:
            Converted 16 kHz waveform as a float32 numpy array of shape
            ``(samples,)``.
        """
        query_seq = self.get_features(speech)
        matching_set = self.get_matching_set(reference_speech, cache_key=target_speaker)
        converted = match_features(query_seq, matching_set, topk=self.topk)
        wav = self.vocode(converted).cpu()
        wav = self._normalize_loudness(wav)
        return wav.numpy().astype(np.float32)
