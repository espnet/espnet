# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallel multimodal LLM implementation for HuggingFace models."""

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from espnet2.speechlm.model.speechlm.lm.loss import fused_cross_entropy_loss


def ParallelHFModel(model_hf_tag, **kwargs):
    """Factory function to create a parallel multimodal LLM from HuggingFace model.

    Args:
        model_hf_tag: HuggingFace model identifier
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Instantiated parallel LLM model with multimodal capabilities
    """
    model_class = build_parallel_hf_class(model_hf_tag)
    return model_class.from_pretrained(model_hf_tag, **kwargs)


def build_parallel_hf_class(model_hf_tag):
    """Dynamically create a parallel LLM class based on HuggingFace architecture.

    Creates a subclass of the original HF model with added multimodal support,
    parallel stream processing, and custom embedding/loss computation.

    Args:
        model_hf_tag: HuggingFace model identifier to determine base architecture

    Returns:
        ParallelLLM class inheriting from the original HF architecture
    """

    config = AutoConfig.from_pretrained(model_hf_tag)
    architecture = config.architectures[0]
    architecture = getattr(transformers, architecture)

    class ParallelLLM(architecture):
        """Parallel multimodal LLM supporting multi-stream token processing."""

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            multimodal_io,
            vocab_meta,
            **kwargs,
        ):
            """Load pretrained model and adapt it for multimodal parallel processing.

            Args:
                pretrained_model_name_or_path: HF model path or identifier
                multimodal_io: Dict of IO handlers for different modalities
                vocab_meta: Dict with vocab, intervals, weights, and size info
                **kwargs: Additional HF model loading arguments

            Returns:
                Model with rebuilt embeddings and multimodal components
            """
            # (1) Load the base model using parent's from_pretrained
            tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
            z_loss_weight = kwargs.pop("z_loss_weight", 0.0)

            model = super(ParallelLLM, cls).from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

            model.z_loss_weight = z_loss_weight

            # (1.5) Assert flash attention is used — our attn_args pre-compute
            # cu_seqlens (pack) or attention_mask (bucket) for flash attention.
            attn_impl = getattr(model.config, "_attn_implementation", "")
            assert "flash_attention" in attn_impl, (
                f"OpusLM requires Flash Attention "
                f"(got attn_implementation={attn_impl!r}). "
                f"Set attn_implementation: flash_attention_2 or "
                f"flash_attention_3 in model_conf."
            )

            # (2) Rebuild embedding tables for multimodal vocabulary
            with torch.no_grad():
                # (2.1) init new embedding and lm head
                vocab_size = vocab_meta["vocab_size"]
                embed_dim = model.config.hidden_size
                new_embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

                nn.init.normal_(new_embed_tokens.weight, mean=0.0, std=0.02)
                nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)

                # (2.2) override by old embedding and lm head
                assert hasattr(model.model, "embed_tokens")
                assert hasattr(model, "lm_head")
                text_start, text_end = vocab_meta["text_start"], vocab_meta["text_end"]

                old_embed = model.model.embed_tokens
                old_lm_head = model.lm_head
                orig_vocab_size = old_embed.weight.shape[0]

                # Validate text vocabulary size matches pretrained model
                if text_end - text_start != orig_vocab_size:
                    raise ValueError(
                        f"text_end - text_start ({text_end - text_start}) "
                        f"must equal original vocab size ({orig_vocab_size})"
                    )

                # Copy pretrained weights to corresponding positions
                new_embed_tokens.weight[text_start:text_end] = old_embed.weight
                new_lm_head.weight[text_start:text_end] = old_lm_head.weight

                model.model.embed_tokens = new_embed_tokens
                if tie_word_embeddings:
                    new_lm_head.weight = new_embed_tokens.weight
                model.lm_head = new_lm_head

            # (3) num_stream and stream_emb
            model.num_stream = vocab_meta["num_stream"]
            model.stream_emb = nn.Embedding(model.num_stream, embed_dim)
            nn.init.zeros_(model.stream_emb.weight)

            # (4) multimodal_vocab_range, vocab_weight
            model.multimodal_vocab_range = (
                vocab_meta["mm_start"],
                vocab_meta["mm_end"],
            )
            model.register_buffer("vocab_weight", vocab_meta["vocab_weight"])

            model.vocab = vocab_meta["vocab"]
            model.vocab_intervals = vocab_meta["vocab_intervals"]

            # (4) Setup multimodal IO handlers and adaptors
            # Discrete IOs use vocabulary, continuous IOs need linear adaptors
            model.multimodal_io_dict = nn.ModuleDict(multimodal_io)
            model.adaptor = nn.ModuleDict()
            for io_name, io in model.multimodal_io_dict.items():
                if not io.is_discrete:
                    model.adaptor[io_name] = nn.Linear(
                        io.feature_dim(),
                        model.config.hidden_size,
                    )

            return model

        def forward(self, **kwargs):
            """Forward pass with optional loss computation.

            Args:
                **kwargs: Dict containing:
                    - seqs: Input token sequences [batch, seq_len, num_streams]
                    - conti_feats: Dict of continuous features by modality
                    - loss_masks: Optional loss weight masks
                    - position_ids: Optional position encodings
                    - attn_args: Pre-computed flash attention kwargs. Pack
                      mode: cu_seq_lens_q/k and max_length_q/k (avoids
                      per-layer .item() CPU-GPU sync). Bucket mode: empty.
                    - past_key_values: Optional KV cache for generation

            Returns:
                Dict with either:
                    - loss and stats (training mode with loss_mask)
                    - logits (inference mode without loss_mask)
            """
            input_ids = kwargs["seqs"]
            loss_mask = kwargs.get("loss_masks")
            position_ids = kwargs.get("position_ids", None)
            attn_args = kwargs.get("attn_args", {})

            inputs_embeds = self._embed(input_ids, kwargs)

            # Forward through base transformer model.
            # attn_args carries pre-computed flash attention kwargs:
            #   pack mode:   cu_seq_lens_q/k, max_length_q/k
            #                (avoids per-layer .item() sync)
            #   bucket mode: empty (flash attention uses is_causal=True)
            output = self.model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                output_router_logits=True,
                use_cache=False,
                **attn_args,
            )

            model_output = (
                output.last_hidden_state,
                getattr(output, "router_logits", None),
            )
            target = (input_ids, loss_mask)
            scale = kwargs.get("loss_scale", None)
            return self._loss(model_output, target, scale=scale)

        def reset_loss_stats(self):
            """Reset accumulated loss statistics before a new forward pass.

            Must be called before _loss() to start fresh accumulation.
            In PP mode, call this before schedule.step() so that stats from
            all microbatches accumulate correctly.
            """
            self._loss_stats = {}

        def _loss(self, model_output, target, scale=None):
            """Compute loss from transformer output.

            Applies stream embeddings, fused cross-entropy loss, and optional
            MoE load balancing loss. Accumulates stats into
            self._loss_stats so that the PP schedule — which only
            propagates the returned loss tensor for backward — can still
            access them after step(). Call reset_loss_stats() before the
            first _loss() call in a step to start fresh.

            Args:
                model_output: Tuple of (last_hidden_state, router_logits).
                    last_hidden_state: [batch, seq_len, hidden_dim].
                    router_logits: Tuple of per-layer router logits, or None.
                target: Tuple of (input_ids, loss_mask).
                    input_ids: [batch, seq_len, num_streams] (unshifted;
                        next-token shift is handled inside
                        fused_cross_entropy_loss).
                    loss_mask: [batch, seq_len, num_streams].
                scale: Optional loss scale factor. In PP mode, the trainer
                    pre-computes dp_size / (global_count * n_microbatches)
                    and passes it here so the schedule's backward uses the
                    correctly normalized loss.

            Returns:
                Scalar loss tensor (for backward).
                Side-effects: accumulates into self._loss_stats.
            """
            last_hidden_state, router_logits = model_output
            input_ids, loss_mask = target

            # Add stream embeddings to create stream-specific representations
            # Shape: [batch, seq, hidden] -> [batch, seq, streams, hidden]
            hidden_states = last_hidden_state.unsqueeze(2)

            # Convert DTensor to regular tensor for in-place operations
            stream_weight = self.stream_emb.weight
            if hasattr(stream_weight, "full_tensor"):
                stream_weight = stream_weight.full_tensor()

            stream_weight = stream_weight[None, None, :, :].clone()
            stream_weight[:, :, 0] = 0.0  # First stream uses base representation
            hidden_states = hidden_states + stream_weight.to(hidden_states.dtype)

            ce_loss, count, stats = fused_cross_entropy_loss(
                hidden_states,
                input_ids,
                loss_mask,
                self.lm_head.weight,
                self.multimodal_vocab_range,
                self.num_stream,
                self.training,
                z_loss_weight=getattr(self, "z_loss_weight", 0.0),
                ce_weight=self.vocab_weight,
            )

            stats["count"] = count.detach()
            stats["ce_loss"] = ce_loss.clone().detach()

            # MoE load balance loss (scale to raw-sum space to match ce_loss)
            if router_logits is not None and hasattr(self, "load_balancing_loss_func"):
                aux_loss = self.load_balancing_loss_func(
                    router_logits,
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                )
                loss = ce_loss + aux_loss * count * self.config.router_aux_loss_coef
                stats["load_balance_loss"] = (aux_loss * count).detach()
            else:
                loss = ce_loss

            if scale is not None:
                loss = loss * scale

            for k, v in stats.items():
                v = v.detach() if isinstance(v, torch.Tensor) else v
                if k not in self._loss_stats:
                    self._loss_stats[k] = v
                else:
                    self._loss_stats[k] = self._loss_stats[k] + v

            return loss

        def _embed(self, input_ids, kwargs):
            """Create embeddings from multimodal inputs.

            Handles both discrete tokens (encoded on-the-fly) and continuous
            features (projected through adaptors). Updates input_ids in-place
            for discrete features and builds final embeddings.

            Args:
                input_ids: Token sequences to be embedded [batch, seq_len, streams]
                conti_feats: Dict of features by modality (both discrete & continuous)

            Returns:
                Combined embeddings [batch, seq_len, hidden_dim]
            """

            # (1) Process discrete modalities: encode and place tokens
            for io_name in self.multimodal_io_dict:
                if not self.multimodal_io_dict[io_name].is_discrete:
                    continue

                if (
                    f"{io_name}_indices" not in kwargs
                    or f"{io_name}_feats" not in kwargs
                    or f"{io_name}_lengths" not in kwargs
                ):
                    continue

                # Encode features to discrete codes
                io_indices = kwargs[f"{io_name}_indices"]
                io_feats = kwargs[f"{io_name}_feats"]
                io_lengths = kwargs[f"{io_name}_lengths"]
                codes = self.multimodal_io_dict[io_name].encode_batch(
                    io_feats, io_lengths
                )
                # Add vocabulary offset for this modality
                codes = codes + self.vocab_intervals[io_name][0][0]
                # Place codes in correct positions
                for code, (bidx, start, length) in zip(codes, io_indices):
                    input_ids[bidx, start : start + length] = code[:length]

            # (2) Convert tokens to embeddings and sum across streams
            # NOTE(Jinchuan): Padding tokens in stream > 0 are zeroed out.
            # Cannot do the same for stream 0 in Qwen3 for numerical stability.
            input_embeds = self.model.embed_tokens(input_ids)
            input_embeds[..., 1:, :] = torch.where(
                (input_ids[..., 1:] == 0).unsqueeze(-1), 0.0, input_embeds[..., 1:, :]
            )
            input_embeds = input_embeds.sum(dim=2)

            # (3) Process continuous modalities: encode and project features
            for io_name in self.multimodal_io_dict:
                if self.multimodal_io_dict[io_name].is_discrete:
                    continue
                if (
                    f"{io_name}_indices" not in kwargs
                    or f"{io_name}_feats" not in kwargs
                    or f"{io_name}_lengths" not in kwargs
                ):
                    continue

                # Encode features to discrete codes
                io_indices = kwargs[f"{io_name}_indices"]
                io_feats = kwargs[f"{io_name}_feats"]
                io_lengths = kwargs[f"{io_name}_lengths"]
                io_feats = self.multimodal_io_dict[io_name].encode_batch(
                    io_feats, io_lengths
                )
                for feat, (bidx, start, length) in zip(io_feats, io_indices):
                    feat = self.adaptor[io_name](feat)
                    # NOTE(Jinchuan): Force the length to match
                    input_embeds[bidx, start : start + length] = feat

            # (4) Add dummy forward to ensure all multimodal_io are always included
            # in the computation graph, even if not used in this batch.
            # This prevents gradient mismatch errors in DeepSpeed ZeRO.
            for io_name in self.multimodal_io_dict:
                if io_name == "text":
                    continue

                # Skip if this modality was already used in this batch
                if (
                    f"{io_name}_feats" in kwargs
                    and kwargs[f"{io_name}_feats"] is not None
                    and len(kwargs[f"{io_name}_feats"]) > 0
                ):
                    continue

                # Use dummy_forward from the IO class
                io_module = self.multimodal_io_dict[io_name]
                dummy_out = io_module.dummy_forward(ref_tensor=input_embeds)

                # For continuous modalities, also run through adaptor
                if not io_module.is_discrete and io_name in self.adaptor:
                    dummy_out = self.adaptor[io_name](dummy_out)
                dummy_out = dummy_out.to(input_embeds.dtype)

                # Sum and add with zero weight to include in computation graph
                input_embeds = input_embeds + 0.0 * dummy_out.sum()

            return input_embeds

        # Below are all inference logics
        @torch.no_grad()
        def inference(self, inference_config: dict, cache: list = None, **kwargs):

            # (1) Prefill input_ids
            input_ids = kwargs.get("seqs")
            input_embeds = self._embed(input_ids, kwargs)

            _, cache = self._step(
                input_embeds=input_embeds,
                past_key_values=cache,
            )

            messages = []
            num_msg = 0
            enforce_modalities = inference_config.get("enforce_modality", [])
            while True:
                # (2.1) Prefill assistant token
                logits, cache = self._step(
                    input_ids=self.assistant_token,
                    past_key_values=cache,
                    mask=self.modality_mask,
                )

                # (2.2) determine modality token and mask
                try:
                    modality = enforce_modalities[num_msg]
                    modality_token = getattr(self, f"{modality}_token")
                except Exception:
                    modality_token = logits.argmax(3)
                    modality = modality_token.flatten()[0].item()
                    modality = self.vocab[modality].replace("<|", "").replace("|>", "")
                modality_mask = getattr(self, f"{modality}_mask")

                # (2.3) predict token sequence
                decoded_sequences, cache, logits = self.inference_segment(
                    config=inference_config[modality],
                    cache=cache,
                    prev_token=modality_token,
                    mask=modality_mask,
                )

                # (2.4) detokenization
                for seq in decoded_sequences:
                    if (
                        seq[-1, 0] == self.eos_token_id
                        or seq[-1, 0] == self.eot_token_id
                    ):
                        seq = seq[:-1]  # remove <|eos|> or <|eou|>

                    io_name = "discrete_audio" if modality == "audio" else modality
                    seq = seq.unsqueeze(0) - self.vocab_intervals[io_name][0][0]

                    io = self.multimodal_io_dict[io_name]
                    lengths = torch.Tensor([seq.size(1)]).long().to(seq.device)
                    content = io.decode_batch(seq, lengths)

                    msg = ["assistant", modality, content]
                    messages.append(msg)

                # (2.5) Terminate when applicable
                if len(decoded_sequences) > 1:
                    break  # multi-segment decoding only supports batch size of 1

                elif (
                    decoded_sequences[0][-1, 0] != self.eot_token_id
                    and num_msg >= len(enforce_modalities) - 1
                ):
                    break  # decode next segment only when ending with <|eot|>

                num_msg += 1

            return messages, cache

        def inference_segment(
            self,
            config: dict,
            cache: list,
            prev_token: torch.Tensor,
            mask: torch.Tensor,
        ):
            device = prev_token.device

            # (1) preprocess for multi-hypothesis inference and CFG
            num_hypo = config.get("num_hypo", 1)
            if num_hypo > 1:
                indices = torch.zeros(num_hypo).long().to(device)
                cache.batch_select_indices(indices)
                prev_token = prev_token.tile(num_hypo, 1, 1)

            cfg = config.get("cfg", 1)
            if cfg > 1:
                cache = self._prepare_cfg_cache(cache)

            # (2) Inference loop
            hypos = list()
            finish_idx = torch.ones(num_hypo).long().to(device) * -1
            for step in range(config["max_step"]):
                # (2.1) Model inference
                if cfg > 1:
                    prev_token = prev_token.tile(2, 1, 1)

                this_mask = mask.clone()
                if step >= config.get("min_step", 1):
                    this_mask[:, :, 0, self.eot_token_id] = False
                    this_mask[:, :, 0, self.eos_token_id] = False
                logits, cache = self._step(
                    input_ids=prev_token, past_key_values=cache, mask=this_mask
                )

                if cfg > 1:
                    logits, cfg_logits = logits.chunk(2)
                    logits = logits * cfg + cfg_logits * (1 - cfg)
                    logits.masked_fill_(this_mask, float("-inf"))

                # (2.2) token prediction based on logits
                prev_token = self._logits_to_token(
                    logits,
                    temperature=config["temperature"],
                    topk=config["topk"],
                )
                hypos.append(prev_token)

                # (2.3) Break when proper
                finish_here = torch.logical_and(
                    torch.logical_or(
                        prev_token[:, 0, 0] == self.eot_token_id,
                        prev_token[:, 0, 0] == self.eos_token_id,
                    ),
                    finish_idx == -1,
                )
                finish_idx = torch.where(finish_here, step, finish_idx)

                if torch.all(finish_idx >= 0):
                    break

            # (3) Finalize
            finish_idx = torch.where(finish_idx == -1, step, finish_idx)
            hypos = torch.cat(hypos, dim=1)

            if cfg > 1:
                indices = torch.arange(num_hypo).long().to(device)
                cache.batch_select_indices(indices)

            # NOTE(Jinchuan): Prefill the last token. This is effective only for
            # multi-segment inference with batch size of 1
            prev_token[..., 1:] = 0
            last_logits, cache = self._step(input_ids=prev_token, past_key_values=cache)

            # TODO(Jinchuan): If this is for delay-interleaved audio, we should
            # enforce it to have valid paddings here.

            hypo_lst = list()
            for idx, hypo in zip(finish_idx, hypos):
                hypo = hypo[: idx + 1]
                hypo_lst.append(hypo)

            return hypo_lst, cache, last_logits

        def prepare_inference(self):
            # (1) the special tokens for prefill
            tokens = ["assistant", "audio", "text", "eos", "eot"]
            for token in tokens:
                token_id = self.vocab.index(f"<|{token}|>")
                token_tensor = torch.zeros((1, 1, self.num_stream)).long()
                token_tensor[0, 0, 0] = token_id
                self.register_buffer(f"{token}_token", token_tensor)

            # (2) modality mask for modality prediction
            tokens = ["audio", "text", "image", "video", "toolcall"]
            mask = torch.ones(self.num_stream, len(self.vocab)).bool()
            for token in tokens:
                token_id = self.vocab.index(f"<|{token}|>")
                mask[0, token_id] = False
            mask[1:, 0] = False
            mask = mask[None, None, :, :]
            self.register_buffer("modality_mask", mask)

            # (3) mask for restricted decoding for each modality
            self.eot_token_id = self.vocab.index("<|eot|>")
            self.eos_token_id = self.vocab.index("<|eos|>")
            for io_name, intervals in self.vocab_intervals.items():
                mask = torch.ones(self.num_stream, len(self.vocab)).bool()
                for idx, (start, end) in enumerate(intervals):
                    mask[idx, start:end] = False
                for idx in range(len(intervals), self.num_stream):
                    mask[idx, 0] = False  # unused stream: only allow paddings
                # mask[0, self.eot_token_id] = False
                # mask[0, self.eos_token_id] = False

                io_name = "audio" if io_name == "discrete_audio" else io_name
                mask = mask[None, None, :, :]
                self.register_buffer(f"{io_name}_mask", mask)

        def _step(
            self, input_ids=None, input_embeds=None, past_key_values=None, mask=None
        ):

            assert (input_ids is None) != (
                input_embeds is None
            ), "Either input_ids or input_embeds should be None"

            if input_ids is not None:
                assert input_ids.size(2) == self.num_stream
                input_embeds = self.model.embed_tokens(input_ids)
                input_embeds[..., 1:, :] = torch.where(
                    (input_ids[..., 1:] == 0).unsqueeze(-1),
                    0.0,
                    input_embeds[..., 1:, :],
                )
                input_embeds = input_embeds.sum(dim=2)

            output = self.model(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = output.past_key_values
            hidden_states = output.last_hidden_state.unsqueeze(2)
            stream_emb = self.stream_emb.weight[None, None, :, :].clone()
            stream_emb[:, :, 0] = 0.0  # First stream uses base representation
            hidden_states = hidden_states + stream_emb
            logits = self.lm_head(hidden_states)

            if mask is not None:
                logits.masked_fill_(mask, float("-inf"))

            return logits, past_key_values

        def _logits_to_token(self, logits, temperature, topk):
            if temperature == 0:  # greedy
                return logits.argmax(-1)
            else:
                topk_values, topk_indices = torch.topk(logits, topk)
                probs = torch.softmax(topk_values / temperature, dim=-1)
                inner_indices = torch.multinomial(
                    probs.flatten(end_dim=-2), num_samples=1
                ).view(probs[..., :1].size())
                return torch.gather(topk_indices, -1, inner_indices).squeeze(-1)

        def _prepare_cfg_cache(self, cache):
            assert isinstance(cache, DynamicCache)

            device = cache.layers[0].keys.device
            length = cache.get_seq_length()
            batch_size = cache.layers[0].keys.shape[0]

            zeros = torch.zeros((batch_size, length, self.num_stream))
            zeros = zeros.to(device).long()

            _, cfg_cache = self._step(input_ids=zeros)

            combined_cache = DynamicCache()
            for idx in range(len(cache.layers)):
                key = torch.cat(
                    [
                        cache.layers[idx].keys,
                        cfg_cache.layers[idx].keys,
                    ],
                    dim=0,
                )
                value = torch.cat(
                    [
                        cache.layers[idx].values,
                        cfg_cache.layers[idx].values,
                    ],
                    dim=0,
                )
                combined_cache.update(
                    key_states=key,
                    value_states=value,
                    layer_idx=idx,
                )

            return combined_cache

    return ParallelLLM
