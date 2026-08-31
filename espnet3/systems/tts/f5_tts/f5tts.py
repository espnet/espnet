"""F5-TTS as an ESPnet3 model.

ESPnet3 reaches a model through ``model._target_`` and leaves ``task:`` unset,
so the only contract is ``torch.nn.Module`` plus::

    forward(...) -> (loss, stats, weight)
    collect_feats(...) -> {"feats": ..., "feats_lengths": ...}

This class implements both directly: it owns its mel front end
(:class:`VocoderMelSpec`, so training features stay bit-compatible with the
neural vocoder) and the ported flow-matching stack (DiT backbone + conditional
flow-matching objective in ``cfm`` / ``dit``). ``CommonCollateFn`` pads text with
id 0, whereas F5's text embedding expects its filler/padding id to be -1; padded
positions are remapped from ``text_lengths`` before the flow-matching forward.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.amp import autocast

from espnet3.systems.tts.f5_tts.cfm import CFM
from espnet3.systems.tts.f5_tts.dit import DiT
from espnet3.systems.tts.f5_tts.vocoder_mel import VocoderMelSpec
from espnet3.utils.config_utils import convert_to_dict


class F5TTS(torch.nn.Module):
    """F5-TTS flow-matching model, mel front end included."""

    def __init__(
        self,
        token_list: Union[str, Sequence[str]],
        feats_extract_config: Optional[Dict[str, Any]] = None,
        # --- DiT backbone (F5TTS_Base defaults) ---
        hidden_size: int = 1024,
        depth: int = 22,
        attention_heads: int = 16,
        attention_head_size: int = 64,
        feed_forward_multiplier: int = 2,
        text_embedding_size: Optional[int] = 512,
        convolution_layers: int = 4,
        dropout: float = 0.1,
        query_key_normalization: Optional[str] = None,
        text_mask_padding: bool = True,
        rotary_attention_heads: Optional[int] = None,
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
        # --- conditional flow matching ---
        sigma: float = 0.0,
        audio_drop_probability: float = 0.3,
        condition_drop_probability: float = 0.2,
        mask_fraction_range: Tuple[float, float] = (0.7, 1.0),
        ode_solver_method: str = "euler",
        mel_spectrogram_kwargs: Optional[dict] = None,
    ):
        """Build the mel front end, the DiT backbone and the flow-matching wrap.

        Hyper-parameters are spelled out rather than carrying upstream F5-TTS's
        abbreviations. To port a published F5-TTS config (``F5TTS_Base.yaml`` /
        ``F5TTS_Small.yaml``, or the paper's Table 3), translate with:

        ===============================  ==============================
        Upstream F5-TTS                  This class
        ===============================  ==============================
        ``dim``                          ``hidden_size``
        ``heads``                        ``attention_heads``
        ``dim_head``                     ``attention_head_size``
        ``ff_mult``                      ``feed_forward_multiplier``
        ``text_dim``                     ``text_embedding_size``
        ``conv_layers``                  ``convolution_layers``
        ``qk_norm``                      ``query_key_normalization``
        ``pe_attn_head``                 ``rotary_attention_heads``
        ``audio_drop_prob``              ``audio_drop_probability``
        ``cond_drop_prob``               ``condition_drop_probability``
        ``frac_lengths_mask``            ``mask_fraction_range``
        ``odeint_kwargs["method"]``      ``ode_solver_method``
        ``mel_spec_kwargs``              ``mel_spectrogram_kwargs``
        ===============================  ==============================

        ``depth``, ``dropout``, ``sigma``, ``text_mask_padding``,
        ``long_skip_connection`` and ``checkpoint_activations`` keep their
        upstream names, which were already unabbreviated.

        Args:
            token_list: Path to the token file, or the token list itself. Its
                length becomes the vocabulary size.
            feats_extract_config: Keyword arguments for :class:`VocoderMelSpec`.
                Its ``n_mels`` becomes the model's mel dimension, so the recipe
                never states that dimension twice.
            hidden_size: Width of the DiT residual stream.
            depth: Number of DiT blocks.
            attention_heads: Attention heads per block.
            attention_head_size: Width of one attention head.
            feed_forward_multiplier: Feed-forward width as a multiple of
                ``hidden_size``.
            text_embedding_size: Width of the text embedding; ``None`` falls
                back to the mel dimension.
            convolution_layers: ConvNeXt layers in the text encoder.
            dropout: Dropout probability inside the backbone.
            query_key_normalization: ``None`` or ``"rms_norm"``.
            text_mask_padding: Mask padded text positions in the text encoder.
            rotary_attention_heads: How many leading attention heads get rotary
                position embeddings. ``None`` applies them to every head.
            long_skip_connection: Add the DiT long skip connection.
            checkpoint_activations: Trade compute for memory via activation
                checkpointing.
            sigma: Flow-matching noise scale.
            audio_drop_probability: Probability of dropping the audio condition
                during training, for classifier-free guidance.
            condition_drop_probability: Probability of dropping both conditions
                together, for classifier-free guidance.
            mask_fraction_range: ``(min, max)`` fraction of each utterance to
                mask for prediction, drawn uniformly per sample.
            ode_solver_method: ``"euler"`` or ``"midpoint"``.
            mel_spectrogram_kwargs: Settings for the mel the flow-matching
                wrapper extracts when it is conditioned on a raw waveform.

        Raises:
            RuntimeError: If ``token_list`` is neither a path nor a sequence.

        Example:
            .. code-block:: yaml

                model:            # F5TTS_Small; omit the sizes for F5TTS_Base
                  _target_: espnet3.systems.tts.f5_tts.f5tts.F5TTS
                  token_list: ${data_dir}/tokens/char_tokens.txt
                  feats_extract_config:
                    fs: 24000
                    n_fft: 1024
                    hop_length: 256
                    win_length: 1024
                    n_mels: 100
                    mel_spec_type: vocos
                  hidden_size: 768
                  depth: 18
                  attention_heads: 12
                  mask_fraction_range: [0.7, 1.0]

        Note:
            Changing the backbone sizes changes the parameter shapes, so an
            existing checkpoint will not load into a resized model.
        """
        super().__init__()

        tokens = self._load_token_list(token_list)
        self.feats_extract = VocoderMelSpec(
            **(convert_to_dict(feats_extract_config) or {})
        )
        self.mel_dim = self.feats_extract.output_size

        # Left of each ``=`` is the upstream F5-TTS name; see the mapping table
        # in the docstring above.
        backbone = DiT(
            dim=hidden_size,
            depth=depth,
            heads=attention_heads,
            dim_head=attention_head_size,
            ff_mult=feed_forward_multiplier,
            mel_dim=self.mel_dim,
            text_num_embeds=len(tokens),
            text_dim=text_embedding_size,
            text_mask_padding=text_mask_padding,
            qk_norm=query_key_normalization,
            conv_layers=convolution_layers,
            pe_attn_head=rotary_attention_heads,
            long_skip_connection=long_skip_connection,
            checkpoint_activations=checkpoint_activations,
            dropout=dropout,
        )
        self.cfm = CFM(
            transformer=backbone,
            sigma=sigma,
            audio_drop_prob=audio_drop_probability,
            cond_drop_prob=condition_drop_probability,
            num_channels=self.mel_dim,
            mel_spec_kwargs=convert_to_dict(mel_spectrogram_kwargs) or {},
            frac_lengths_mask=tuple(mask_fraction_range),
            odeint_kwargs=dict(method=ode_solver_method),
        )

    @staticmethod
    def _load_token_list(token_list: Union[str, Sequence[str]]) -> List[str]:
        """Read the token list from a file path, or accept it inline."""
        token_list = convert_to_dict(token_list)
        if isinstance(token_list, str):
            with open(token_list, encoding="utf-8") as token_file:
                return [line[0] + line[1:].rstrip() for line in token_file]
        if isinstance(token_list, (tuple, list)):
            return list(token_list)
        raise RuntimeError("token_list must be a path or a sequence of tokens")

    @staticmethod
    def _remap_text_padding(text: torch.Tensor, text_lengths: torch.Tensor):
        """Set padded token positions to F5's filler id (-1).

        ``CommonCollateFn`` pads text with 0, but F5's text embedding adds 1 and
        treats id 0 (i.e. original -1) as the filler/padding token. We rely on
        ``text_lengths`` rather than the pad value so any collate setting works.
        """
        maxlen = text.size(1)
        padding_mask = (
            torch.arange(maxlen, device=text.device)[None, :] >= text_lengths[:, None]
        )
        return text.masked_fill(padding_mask, -1)

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Waveform -> mel, always in fp32 so AMP cannot perturb the front end."""
        with autocast("cuda", enabled=False):
            return self.feats_extract(speech, speech_lengths)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Flow-matching training/validation step.

        Args:
            text: Token ids ``[B, T_text]`` (padded with 0 by CommonCollateFn).
            text_lengths: ``[B]``.
            speech: Waveform batch ``[B, T_wav]``.
            speech_lengths: ``[B]`` valid sample counts.
            **kwargs: Extra batch fields this model does not use.

        Returns:
            Tuple of ``(loss, stats, weight)``, the ESPnet3 training contract: a
            scalar flow-matching loss, ``{"loss": detached loss}`` for logging,
            and ``None`` for the weight.

        Example:
            .. code-block:: python

                >>> loss, stats, weight = model(
                ...     text=text, text_lengths=text_lengths,
                ...     speech=speech, speech_lengths=speech_lengths,
                ... )

        Note:
            The span masked for prediction is drawn at random each call
            (``mask_fraction_range``), so the loss is stochastic: two calls on the
            same batch differ unless the RNG is reseeded.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        text = self._remap_text_padding(text, text_lengths)
        loss, _cond, _pred = self.cfm(feats, text=text, lens=feats_lengths)
        stats = dict(loss=loss.detach())
        return loss, stats, None

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return the mel features for the stats-collection stage.

        Args:
            text: Token ids ``[B, T_text]``. Unused, part of the contract.
            text_lengths: ``[B]``. Unused, part of the contract.
            speech: Waveform batch ``[B, T_wav]``.
            speech_lengths: ``[B]`` valid sample counts.
            **kwargs: Extra batch fields this model does not use.

        Returns:
            ``{"feats": mel[B, T, mel_dim], "feats_lengths": [B]}``.

        Example:
            .. code-block:: python

                >>> sorted(model.collect_feats(
                ...     text=text, text_lengths=text_lengths,
                ...     speech=speech, speech_lengths=speech_lengths,
                ... ))
                ['feats', 'feats_lengths']

        Note:
            ``espnet3.components.data.collect_stats`` calls this to write the
            ``feats_shape`` files the length-based batch sampler reads. F5 needs
            no feature normalization, so only the shapes are used.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return dict(feats=feats, feats_lengths=feats_lengths)

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        duration: Optional[int] = None,
        steps: int = 32,
        guidance_strength: float = 2.0,
        sway_sampling_coefficient: float = -1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate mel for ``text`` given a reference, the minimal path.

        F5 is zero-shot: it needs a reference. ``speech`` is the reference mel or
        waveform; ``text`` should be the (ref + target) token ids, and for the full
        cross-speaker recipe protocol use
        ``espnet3.systems.tts.f5_tts.inference.F5TTSInference``,
        which handles reference pairing and vocoding.

        Args:
            text: Reference + target token ids ``[T_text]``, unbatched.
            speech: Reference audio, either mel ``[T_ref, mel_dim]`` or raw
                waveform ``[T_wav]``. Required.
            duration: Total mel length to generate, reference included. Defaults
                to twice the reference length.
            steps: Number of ODE solver steps.
            guidance_strength: Classifier-free guidance scale
                (upstream ``cfg_strength``).
            sway_sampling_coefficient: Sway sampling coefficient for timestep
                schedule; negative values front-load the steps.
            **kwargs: Ignored.

        Returns:
            ``{"feat_gen": mel[T_gen, mel_dim]}``, with the reference prefix
            stripped so only the generated span is returned.

        Raises:
            RuntimeError: If ``speech`` is ``None``. F5 is zero-shot and cannot
                generate without a reference.

        Example:
            .. code-block:: python

                >>> out = model.inference(text=token_ids, speech=ref_mel)
                >>> out["feat_gen"].shape
                torch.Size([T_gen, 100])

        Note:
            This path does no vocoding. The default ``duration`` heuristic is a
            fallback, and output length is driven by the reference length, so
            use ``F5TTSInference`` for the recipe protocol.
        """
        if speech is None:
            raise RuntimeError("F5TTS.inference requires a reference 'speech'.")

        cond = speech.unsqueeze(0) if speech.dim() <= 2 else speech
        if cond.dim() == 2:  # raw waveform [1, T_wav] -> CFM extracts mel
            reference_length = None
        else:  # mel [1, n, d]
            reference_length = cond.shape[1]

        token_ids = text.unsqueeze(0)
        if duration is None:
            # Fall back to twice the reference length when no estimate is given.
            base = (
                reference_length if reference_length is not None else token_ids.shape[1]
            )
            duration = int(base * 2)

        out, _ = self.cfm.sample(
            cond=cond,
            text=token_ids,
            duration=duration,
            steps=steps,
            cfg_strength=guidance_strength,
            sway_sampling_coef=sway_sampling_coefficient,
        )
        prefix_length = reference_length if reference_length is not None else 0
        return {"feat_gen": out[0, prefix_length:, :]}
