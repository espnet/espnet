"""F5-TTS as an ``AbsTTS`` model.

Implements espnet2's ``AbsTTS`` interface so it can sit inside
``ESPnetTTSModel``, but it is built by ``builder.build_f5_tts_model`` through
Hydra rather than registered in ``espnet2/tasks/tts.py``. The heavy lifting
(DiT backbone + conditional flow-matching objective) lives in the
ported ``cfm`` / ``backbones.dit`` modules; this class only adapts them to the
``AbsTTS`` contract:

    forward(text, text_lengths, feats, feats_lengths, ...) -> (loss, stats, weight)

``ESPnetTTSModel`` extracts ``feats`` (mel) via the configured ``feats_extract``
(``VocoderMelSpec``, to stay vocoder-compatible) and passes them in, so F5TTS works
on precomputed mel here. ``CommonCollateFn`` pads text with id 0, whereas F5's
text embedding expects its filler/padding id to be -1; we remap padded positions
from ``text_lengths`` before the flow-matching forward.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from espnet2.tts.abs_tts import AbsTTS
from espnet3.systems.tts.f5_tts.backbones.dit import DiT
from espnet3.systems.tts.f5_tts.cfm import CFM


class F5TTS(AbsTTS):
    """F5-TTS flow-matching model adapted to the espnet2 TTS interface."""

    def __init__(
        self,
        idim: int,
        odim: int,
        # --- DiT backbone (F5TTS_Base defaults) ---
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: int = 2,
        text_dim: Optional[int] = 512,
        conv_layers: int = 4,
        dropout: float = 0.1,
        qk_norm: Optional[str] = None,
        text_mask_padding: bool = True,
        pe_attn_head: Optional[int] = None,
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
        # --- conditional flow matching ---
        sigma: float = 0.0,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.0),
        odeint_method: str = "euler",
        mel_spec_kwargs: Optional[dict] = None,
    ):
        """Build the DiT backbone and wrap it in the flow-matching objective.

        Args:
            idim: Vocabulary size.
            odim: Mel dimension, supplied by the feature extractor.
            dim, depth, heads, dim_head, ff_mult, text_dim, conv_layers,
                dropout, qk_norm, text_mask_padding, pe_attn_head,
                long_skip_connection, checkpoint_activations: DiT backbone
                hyper-parameters, defaulting to F5TTS_Base.
            sigma, audio_drop_prob, cond_drop_prob, frac_lengths_mask,
                odeint_method, mel_spec_kwargs: Conditional flow-matching
                hyper-parameters forwarded to ``CFM``.

        Example:
            .. code-block:: yaml

                tts_conf:      # F5TTS_Small; omit these keys for F5TTS_Base
                  dim: 768
                  depth: 18
                  heads: 12
                  frac_lengths_mask: [0.7, 1.0]

        Note:
            ``idim`` and ``odim`` are supplied by
            ``builder.build_f5_tts_model``: ``idim`` is the token-list length
            and ``odim`` comes from ``VocoderMelSpec.output_size``. Every
            other argument is a key of the recipe's ``tts_conf`` block. Changing
            the backbone sizes changes the parameter shapes, so an existing
            checkpoint will not load into a resized model.
        """
        super().__init__()
        self.odim = odim

        backbone = DiT(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            mel_dim=odim,
            text_num_embeds=idim,
            text_dim=text_dim,
            text_mask_padding=text_mask_padding,
            qk_norm=qk_norm,
            conv_layers=conv_layers,
            pe_attn_head=pe_attn_head,
            long_skip_connection=long_skip_connection,
            checkpoint_activations=checkpoint_activations,
            dropout=dropout,
        )
        self.cfm = CFM(
            transformer=backbone,
            sigma=sigma,
            audio_drop_prob=audio_drop_prob,
            cond_drop_prob=cond_drop_prob,
            num_channels=odim,
            mel_spec_kwargs=mel_spec_kwargs or {},
            frac_lengths_mask=tuple(frac_lengths_mask),
            odeint_kwargs=dict(method=odeint_method),
        )

    @property
    def require_raw_speech(self) -> bool:
        """Return False: F5 trains on mel, extracted by ``feats_extract``.

        Returns:
            Always ``False``.

        Note:
            ``ESPnetTTSModel`` reads this to decide whether to hand the model
            the raw waveform. F5 wants mel, so ``VocoderMelSpec`` runs first and
            ``forward`` receives ``feats``. ``inference`` still accepts a
            waveform reference, which ``CFM`` converts internally.
        """
        return False

    @property
    def require_vocoder(self) -> bool:
        """Return True: the mel this model produces needs a neural vocoder.

        Returns:
            Always ``True``.

        Note:
            The model emits mel, never a waveform. Which vocoder family that mel
            targets is set by ``VocoderMelSpec(mel_spec_type=...)``, so the
            vocoder used at inference must match the one used for training.
        """
        return True

    @staticmethod
    def _remap_text_padding(text: torch.Tensor, text_lengths: torch.Tensor):
        """Set padded token positions to F5's filler id (-1).

        ``CommonCollateFn`` pads text with 0, but F5's text embedding adds 1 and
        treats id 0 (i.e. original -1) as the filler/padding token. We rely on
        ``text_lengths`` rather than the pad value so any collate setting works.
        """
        maxlen = text.size(1)
        pad_mask = (
            torch.arange(maxlen, device=text.device)[None, :] >= text_lengths[:, None]
        )
        return text.masked_fill(pad_mask, -1)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Flow-matching training/validation step.

        Args:
            text: Token ids ``[B, T_text]`` (padded with 0 by CommonCollateFn).
            text_lengths: ``[B]``.
            feats: Mel spectrogram ``[B, T_feats, odim]`` from feats_extract.
            feats_lengths: ``[B]`` valid mel frame counts.
            **kwargs: Extra batch fields (e.g. ``speech``) that ``ESPnetTTSModel``
                forwards but this model does not use.

        Returns:
            Tuple of ``(loss, stats, weight)``, the ``AbsTTS`` training contract:
            a scalar flow-matching loss, ``{"loss": detached loss}`` for logging,
            and ``None`` for the weight.

        Example:
            .. code-block:: python

                >>> loss, stats, weight = model.tts(
                ...     text=text, text_lengths=text_lengths,
                ...     feats=feats, feats_lengths=feats_lengths,
                ... )

        Note:
            The span masked for prediction is drawn at random each call
            (``frac_lengths_mask``), so the loss is stochastic: two calls on the
            same batch differ unless the RNG is reseeded.
        """
        text = self._remap_text_padding(text, text_lengths)
        loss, _cond, _pred = self.cfm(feats, text=text, lens=feats_lengths)
        stats = dict(loss=loss.detach())
        return loss, stats, None

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        duration: Optional[int] = None,
        steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate mel for ``text`` given a reference, the minimal AbsTTS path.

        F5 is zero-shot: it needs a reference. ``speech`` is the reference mel or
        waveform; ``text`` should be the (ref + target) token ids, and for the full
        cross-speaker recipe protocol use
        ``espnet3.systems.tts.f5_tts.inference.F5TTSInference``,
        which handles reference pairing and vocoding.

        Args:
            text: Reference + target token ids ``[T_text]``, unbatched.
            speech: Reference audio, either mel ``[T_ref, odim]`` or raw
                waveform ``[T_wav]``. Required.
            ref_text: Accepted for interface compatibility and unused; the
                reference transcript is expected to be part of ``text``.
            duration: Total mel length to generate, reference included. Defaults
                to twice the reference length.
            steps: Number of ODE solver steps.
            cfg_strength: Classifier-free guidance scale.
            sway_sampling_coef: Sway sampling coefficient for the timestep
                schedule; negative values front-load the steps.
            **kwargs: Ignored, for ``AbsTTS`` signature compatibility.

        Returns:
            ``{"feat_gen": mel[T_gen, odim]}``, with the reference prefix
            stripped so only the generated span is returned.

        Raises:
            RuntimeError: If ``speech`` is ``None``. F5 is zero-shot and cannot
                generate without a reference.

        Example:
            .. code-block:: python

                >>> out = model.tts.inference(text=token_ids, speech=ref_mel)
                >>> out["feat_gen"].shape
                torch.Size([T_gen, 100])

        Note:
            This is the minimal ``AbsTTS`` path and does no vocoding. The
            default ``duration`` heuristic is a fallback, and output length is
            driven by the reference length, so use ``F5TTSInference`` for the
            recipe protocol.
        """
        if speech is None:
            raise RuntimeError("F5TTS.inference requires a reference 'speech'.")

        cond = speech.unsqueeze(0) if speech.dim() <= 2 else speech
        if cond.dim() == 2:  # raw waveform [1, T_wav] -> CFM extracts mel
            ref_len_known = None
        else:  # mel [1, n, d]
            ref_len_known = cond.shape[1]

        text_ids = text.unsqueeze(0)
        if duration is None:
            # Fall back to twice the reference length when no estimate is given.
            base = ref_len_known if ref_len_known is not None else text_ids.shape[1]
            duration = int(base * 2)

        out, _ = self.cfm.sample(
            cond=cond,
            text=text_ids,
            duration=duration,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        ref_len = ref_len_known if ref_len_known is not None else 0
        return {"feat_gen": out[0, ref_len:, :]}
