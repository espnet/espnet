# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Pipeline-parallel multimodal LLM implementation.

Extends ParallelLLM from parallel.py with pipeline-parallel stage-wise
model construction and forward pass. Each PP rank holds only a subset
of transformer layers plus the peripherals needed for its stage role
(first / middle / last).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from espnet2.speechlm.model.speechlm.lm.parallel import build_parallel_hf_class

logger = logging.getLogger(__name__)


def ParallelPPHFModel(model_hf_tag, **kwargs):
    """Factory function to create a PP-aware parallel multimodal LLM.

    Args:
        model_hf_tag: HuggingFace model identifier
        **kwargs: Must include ``parallel_dims`` and ``pp_layout``.
            Remaining kwargs are forwarded to from_pretrained.

    Returns:
        Instantiated stage-local model with only this rank's modules.
    """
    model_class = build_parallel_pp_hf_class(model_hf_tag)
    return model_class.from_pretrained(model_hf_tag, **kwargs)


def build_parallel_pp_hf_class(model_hf_tag):
    """Create a PP-aware LLM class that inherits from ParallelLLM.

    The returned ``ParallelPPLLM`` overrides ``from_pretrained`` (to load
    only the local stage's weights) and ``forward`` (to run stage-wise
    computation).
    """

    ParallelLLM = build_parallel_hf_class(model_hf_tag)

    class ParallelPPLLM(ParallelLLM):
        """Pipeline-parallel multimodal LLM.

        Each instance represents one PP stage. Inherits _loss, _embed,
        reset_loss_stats, and all inference methods from ParallelLLM.
        """

        # ----------------------------------------------------------------
        # from_pretrained: load / prune / broadcast
        # ----------------------------------------------------------------

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            multimodal_io,
            vocab_meta,
            *,
            parallel_dims,
            pp_layout: List[int],
            vpp_index: int = 0,
            **kwargs,
        ):
            """Load a pretrained model and prune it to the local PP stage.

            Only dp_rank == 0 (within the FSDP group) downloads real
            weights. Other ranks create an empty structure with the same
            shape, prune identically, then receive weights via broadcast.

            ``pp_layout`` describes all virtual stages. Its length must
            be a multiple of ``pp_degree``; the ratio gives
            ``vpp_degree`` (virtual stages per rank). Each PP rank holds
            ``vpp_degree`` stages via loop-style assignment: rank *r*
            gets stages ``[r, r + pp_degree, r + 2*pp_degree, ...]``.
            ``vpp_index`` selects which of those this call builds
            (0-based within the rank's set).

            Args:
                pretrained_model_name_or_path: HF model path or identifier.
                multimodal_io: Dict of IO handlers for different modalities.
                vocab_meta: Dict with vocab, intervals, weights, and size info.
                parallel_dims: TorchTitan ParallelDims (must have meshes built).
                pp_layout: List of ints — number of layers per virtual stage.
                    Length must be a multiple of ``pp_degree``.
                vpp_index: Which virtual stage on this rank to build (default 0).
                **kwargs: Additional HF model loading arguments.

            Returns:
                Model pruned to the local stage with weights broadcast.
            """
            z_loss_weight = kwargs.pop("z_loss_weight", 0.0)

            # (0a) PP does not support tied embeddings (embed_tokens on first
            # stage and lm_head on last stage live on different GPUs).
            tie = kwargs.pop("tie_word_embeddings", False)
            assert not tie, (
                "tie_word_embeddings is not supported with pipeline parallelism. "
                "Set tie_word_embeddings: false in model_conf."
            )

            # (0b) PP bypasses HF's causal mask creation and relies on
            # Flash Attention to infer the mask from position_ids.
            attn_impl = kwargs.get("attn_implementation", "")
            assert "flash_attention" in attn_impl, (
                f"Pipeline parallelism requires Flash Attention "
                f"(got attn_implementation={attn_impl!r}). "
                f"Set attn_implementation: flash_attention_2 or "
                f"flash_attention_3 in model_conf."
            )

            # (1) Extract PP / DP topology
            pp_mesh = parallel_dims.get_mesh("pp")
            pp_rank = pp_mesh.get_local_rank()
            pp_degree = pp_mesh.size()

            fsdp_mesh = parallel_dims.get_mesh("fsdp")
            dp_rank = fsdp_mesh.get_local_rank()

            num_virtual_stages = len(pp_layout)
            assert num_virtual_stages % pp_degree == 0, (
                f"pp_layout length ({num_virtual_stages}) must be "
                f"divisible by pp_degree ({pp_degree})"
            )

            # (2) Compute local layer range via loop-style stage assignment.
            # Rank r's virtual stages: [r, r+pp_degree, r+2*pp_degree, ...]
            # vpp_index selects which one this call builds.
            stage_idx = pp_rank + vpp_index * pp_degree
            is_first_stage = stage_idx == 0
            is_last_stage = stage_idx == num_virtual_stages - 1

            layer_start = sum(pp_layout[:stage_idx])
            layer_end = layer_start + pp_layout[stage_idx]

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            total_layers = config.num_hidden_layers
            assert sum(pp_layout) == total_layers, (
                f"sum(pp_layout)={sum(pp_layout)} must equal "
                f"num_hidden_layers={total_layers}"
            )

            logger.info(
                f"PP rank {pp_rank}/{pp_degree}, "
                f"virtual stage {stage_idx}/{num_virtual_stages} "
                f"(vpp_index={vpp_index}): layers [{layer_start}, {layer_end}), "
                f"dp_rank={dp_rank}, first={is_first_stage}, last={is_last_stage}"
            )

            # (3) Load or empty-init the full model
            if dp_rank == 0:
                model = super(ParallelPPLLM, cls).from_pretrained(
                    pretrained_model_name_or_path,
                    multimodal_io,
                    vocab_meta,
                    **kwargs,
                )
            else:
                model = cls._empty_init(
                    pretrained_model_name_or_path,
                    multimodal_io,
                    vocab_meta,
                    **kwargs,
                )

            # (4) Prune to local stage (all ranks)
            cls._prune_to_stage(
                model,
                layer_start,
                layer_end,
                is_first_stage,
                is_last_stage,
            )

            # (5) Move all ranks to GPU then broadcast from dp_rank 0.
            # dp_rank == 0 has real weights on CPU after from_pretrained;
            # dp_rank != 0 has meta tensors after _empty_init + prune.
            # Materialize meta tensors directly on GPU (avoids CPU peak),
            # then move dp_rank 0 to GPU as well. NCCL broadcast requires
            # all tensors on the same CUDA device.
            if dp_rank != 0:
                model.to_empty(device="cuda")
            else:
                model.cuda()

            fsdp_group = fsdp_mesh.get_group()
            src_rank = dist.get_process_group_ranks(fsdp_group)[0]
            for p in model.parameters():
                dist.broadcast(p.data, src=src_rank, group=fsdp_group)
            for name, buf in model.named_buffers():
                dist.broadcast(buf.data, src=src_rank, group=fsdp_group)

            # (6) Store PP metadata and config attributes
            model.pp_rank = pp_rank
            model.pp_degree = pp_degree
            model.is_first_stage = is_first_stage
            model.is_last_stage = is_last_stage
            model.stage_idx = stage_idx
            model.num_virtual_stages = num_virtual_stages
            model.z_loss_weight = z_loss_weight

            return model

        @classmethod
        def _empty_init(
            cls,
            pretrained_model_name_or_path,
            multimodal_io,
            vocab_meta,
            **kwargs,
        ):
            """Create the model structure on meta device (zero CPU memory).

            All parameters are meta tensors (no real storage). After
            pruning removes unneeded modules, the caller materializes
            the remaining parameters to real tensors via broadcast.
            """
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

            with torch.device("meta"):
                hf_model = AutoModelForCausalLM.from_config(config)

            model = hf_model
            model.__class__ = cls

            vocab_size = vocab_meta["vocab_size"]
            embed_dim = config.hidden_size

            with torch.device("meta"):
                new_embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
                model.model.embed_tokens = new_embed_tokens
                model.lm_head = new_lm_head

                model.stream_emb = nn.Embedding(vocab_meta["num_stream"], embed_dim)

            model.num_stream = vocab_meta["num_stream"]
            model.multimodal_vocab_range = (
                vocab_meta["mm_start"],
                vocab_meta["mm_end"],
            )
            model.register_buffer(
                "vocab_weight",
                vocab_meta["vocab_weight"].to(device="meta"),
            )
            model.vocab = vocab_meta["vocab"]
            model.vocab_intervals = vocab_meta["vocab_intervals"]

            # Create multimodal modules with the same structure and shapes
            # as the real ones, but on meta device. This ensures the
            # parameter shapes match for broadcast from dp_rank 0.
            # Non-first stages will have these set to None by
            # _prune_to_stage before broadcast (so no memory cost).
            model.multimodal_io_dict = nn.ModuleDict(multimodal_io)
            model.adaptor = nn.ModuleDict()
            for io_name, io in model.multimodal_io_dict.items():
                if not io.is_discrete:
                    with torch.device("meta"):
                        model.adaptor[io_name] = nn.Linear(
                            io.feature_dim(),
                            config.hidden_size,
                        )

            return model

        @staticmethod
        def _prune_to_stage(
            model,
            layer_start,
            layer_end,
            is_first_stage,
            is_last_stage,
        ):
            """Remove modules not needed by this PP stage in-place.

            Non-local transformer layers are replaced with ``nn.Identity``
            placeholders so the ``ModuleList`` keeps its original indices
            and parameter names retain their global layer numbers.
            These placeholders are skipped in ``_run_decoder_layers``.
            """
            for i in range(len(model.model.layers)):
                if i < layer_start or i >= layer_end:
                    model.model.layers[i] = nn.Identity()

            if not is_first_stage:
                model.model.embed_tokens = None
                model.multimodal_io_dict = None
                model.adaptor = None

            if not is_last_stage:
                model.lm_head = None
                model.stream_emb = None
                model.model.norm = None
                if hasattr(model, "vocab_weight"):
                    model.vocab_weight = None

            num_real = layer_end - layer_start
            logger.info(
                f"Pruned to {num_real} layers "
                f"[{layer_start}:{layer_end}] "
                f"(total slots={len(model.model.layers)}), "
                f"embed={'kept' if is_first_stage else 'removed'}, "
                f"lm_head={'kept' if is_last_stage else 'removed'}"
            )

        # ----------------------------------------------------------------
        # forward: stage-wise computation
        # ----------------------------------------------------------------

        def forward(self, *args, **kwargs):
            """Stage-wise forward pass.

            The PP schedule calls ``forward(*composite_args, **kwargs)``.
            For the first stage, ``composite_args`` is empty and all input
            comes via ``**kwargs``. For subsequent stages,
            ``composite_args`` contains the tensors sent by the previous
            stage.

            Inter-stage output format:
                Dense model:  just ``hidden_states`` (single tensor).
                MoE model:    ``(hidden_states, router_logits)`` (two tensors).
            The receiving stage distinguishes via ``len(args)``.

            First stage:
                Input: **kwargs (batch dict with seqs, feats, masks, ...).
                Output: hidden_states, or (hidden_states, router_logits).

            Middle stage:
                Input: hidden_states [, router_logits] as positional args.
                Output: hidden_states, or (hidden_states, router_logits).

            Last stage:
                Input: hidden_states [, router_logits] as positional args,
                       plus **kwargs with seqs and loss_masks.
                Output: scalar loss tensor.
            """
            if self.pp_degree == 1:
                return super().forward(**kwargs)
            elif self.is_first_stage:
                return self._forward_first_stage(**kwargs)
            elif self.is_last_stage:
                return self._forward_last_stage(args, **kwargs)
            else:
                return self._forward_middle_stage(args, **kwargs)

        def _forward_first_stage(self, **kwargs):
            """First stage: embed -> local layers -> output."""
            input_ids = kwargs["seqs"]
            position_ids = kwargs.get("position_ids", None)
            attn_args = kwargs.get("attn_args", {})

            hidden_states = self._embed(input_ids, kwargs)
            hidden_states, router_logits = self._run_decoder_layers(
                hidden_states,
                position_ids,
                attn_args=attn_args,
            )
            if router_logits is not None:
                return hidden_states, router_logits
            return hidden_states

        def _forward_middle_stage(self, stage_args, **kwargs):
            """Middle stage: local layers on received hidden_states."""
            hidden_states = stage_args[0]
            prev_router_logits = stage_args[1] if len(stage_args) > 1 else None
            position_ids = kwargs.get("position_ids", None)
            attn_args = kwargs.get("attn_args", {})

            hidden_states, local_router_logits = self._run_decoder_layers(
                hidden_states,
                position_ids,
                attn_args=attn_args,
            )
            router_logits = _merge_router_logits(
                prev_router_logits,
                local_router_logits,
            )
            if router_logits is not None:
                return hidden_states, router_logits
            return hidden_states

        def _forward_last_stage(self, stage_args, **kwargs):
            """Last stage: local layers -> norm -> loss.

            Returns scalar loss. The PP schedule uses an identity loss_fn
            so this return value is used directly for backward.
            """
            hidden_states = stage_args[0]
            prev_router_logits = stage_args[1] if len(stage_args) > 1 else None
            position_ids = kwargs.get("position_ids", None)
            attn_args = kwargs.get("attn_args", {})

            hidden_states, local_router_logits = self._run_decoder_layers(
                hidden_states,
                position_ids,
                attn_args=attn_args,
            )
            router_logits = _merge_router_logits(
                prev_router_logits,
                local_router_logits,
            )

            hidden_states = self.model.norm(hidden_states)

            input_ids = kwargs["seqs"]
            loss_mask = kwargs.get("loss_masks")
            scale = kwargs.get("loss_scale", None)

            if router_logits is not None:
                router_logits = (router_logits,)

            model_output = (hidden_states, router_logits)
            target = (input_ids, loss_mask)
            return self._loss(model_output, target, scale=scale)

        # ----------------------------------------------------------------
        # _run_decoder_layers: general layer-wise inference helper
        # ----------------------------------------------------------------

        def _run_decoder_layers(
            self,
            hidden_states: torch.Tensor,
            position_ids: Optional[torch.Tensor],
            attn_args: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Run local decoder layers on hidden_states.

            General helper that works for any HF decoder model whose layers
            accept ``(hidden_states, position_ids=, position_embeddings=)``.

            For MoE models, router_logits are collected from
            ``GroupedMoeBlock._last_router_logits`` after each layer call.
            HF decoder layers discard router_logits internally, so we
            cannot rely on the layer return value.

            Args:
                hidden_states: [batch, seq_len, hidden_dim].
                position_ids: [batch, seq_len] or None.
                attn_args: Pre-computed flash attention kwargs. Pack mode:
                    cu_seq_lens_q/k and max_length_q/k (avoids per-layer
                    .item() sync). Bucket mode: empty.

            Returns:
                (hidden_states, router_logits) where router_logits is a
                concatenated [N, num_experts] tensor for MoE models, or
                None for dense models.
            """
            if attn_args is None:
                attn_args = {}

            position_embeddings = self.model.rotary_emb(
                hidden_states,
                position_ids,
            )

            router_logits_list: List[torch.Tensor] = []
            for layer in self.model.layers:
                if isinstance(layer, nn.Identity):
                    continue

                layer_output = layer(
                    hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    **attn_args,
                )
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

                # HF decoder layers discard router_logits from the MoE
                # block's return value. Retrieve them from the stashed
                # attribute on GroupedMoeBlock instead. Use .clone() to
                # detach from the stash so activation checkpointing
                # recomputation cannot overwrite our copy.
                mlp = getattr(layer, "mlp", None)
                # Unwrap CheckpointWrapper from AC mode="moe"/"moe_and_full"
                if mlp is not None and hasattr(mlp, "_checkpoint_wrapped_module"):
                    mlp = mlp._checkpoint_wrapped_module
                if mlp is not None and hasattr(mlp, "_last_router_logits"):
                    logits = getattr(mlp, "_last_router_logits", None)
                    if logits is not None:
                        router_logits_list.append(logits.clone())
                        mlp._last_router_logits = None

            if router_logits_list:
                return hidden_states, torch.cat(router_logits_list, dim=0)
            return hidden_states, None

    return ParallelPPLLM


def _merge_router_logits(
    prev: Optional[torch.Tensor],
    local: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Concatenate router logits from previous stages with local ones."""
    if prev is None:
        return local
    if local is None:
        return prev
    return torch.cat([prev, local], dim=0)
