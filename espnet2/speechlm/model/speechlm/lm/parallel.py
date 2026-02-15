# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallel multimodal LLM implementation for HuggingFace models."""

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig


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
            vocab_intervals,
            max_loss_interval: int = 13192,
            **kwargs,
        ):
            """Load pretrained model and adapt it for multimodal parallel processing.

            Args:
                pretrained_model_name_or_path: HF model path or identifier
                multimodal_io: Dict of IO handlers for different modalities
                vocab_intervals: Token range mappings for each modality
                max_loss_interval: Maximum interval size for efficient loss computation
                **kwargs: Additional HF model loading arguments

            Returns:
                Model with rebuilt embeddings and multimodal components
            """
            # (1) Load the base model using parent's from_pretrained
            model = super(ParallelLLM, cls).from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

            # (2) Rebuild embedding tables for multimodal vocabulary
            # Strategy: Create new embeddings with unified vocabulary size,
            # preserve text embeddings from pretrained model, initialize
            # others randomly. Token 0 reserved for padding (zero embedding).
            with torch.no_grad():
                # Calculate total vocabulary size across all modalities
                vocab_size = max(
                    [
                        end
                        for intervals in vocab_intervals.values()
                        for _, end in intervals
                    ]
                )

                embed_dim = model.config.hidden_size
                new_embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
                new_embed_tokens.weight[0] = 0.0
                new_lm_head.weight[0] = 0.0

                # Preserve pretrained text embeddings if text modality exists
                if "text" in vocab_intervals:

                    if not (
                        hasattr(model, "model") and hasattr(model.model, "embed_tokens")
                    ):
                        raise AttributeError(
                            "Model must have 'model.embed_tokens' attribute"
                        )
                    if not hasattr(model, "lm_head"):
                        raise AttributeError("Model must have 'lm_head' attribute")

                    text_start, text_end = vocab_intervals["text"][0]

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
                model.lm_head = new_lm_head

            # (3) Create stream embeddings for multi-stream token processing
            # Each stream gets its own embedding offset (except first stream)
            possible_num_stream = [
                io.num_stream() for io in multimodal_io.values() if io.is_discrete
            ]
            if len(possible_num_stream) == 0:
                raise ValueError("Cannot proceed with all IOs being continuous")
            model.num_stream = max(possible_num_stream)
            model.stream_emb = nn.Embedding(model.num_stream, embed_dim)

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

            # (5) Create loss computation intervals for efficient softmax
            # Split large vocabularies into smaller intervals to avoid OOM
            model.vocab_intervals = vocab_intervals
            model.loss_intervals = list()
            for io_name, intervals in vocab_intervals.items():
                # Skip text/special tokens (handled with full softmax in stream 0)
                if io_name == "text" or io_name == "special_token":
                    continue

                cur_start, end = intervals[0]
                # Split intervals if they exceed max_loss_interval size
                for _, end in intervals[1:]:
                    if end - cur_start <= max_loss_interval:
                        continue
                    else:
                        model.loss_intervals.append((cur_start, end))
                        cur_start = end

                # Add final interval if any tokens remain
                if end > cur_start:
                    model.loss_intervals.append((cur_start, end))

            return model

        def forward(self, **kwargs):
            """Forward pass with optional loss computation.

            Args:
                **kwargs: Dict containing:
                    - seqs: Input token sequences [batch, seq_len, num_streams]
                    - conti_feats: Dict of continuous features by modality
                    - loss_masks: Optional loss weight masks
                    - position_ids: Optional position encodings
                    - past_key_values: Optional KV cache for generation

            Returns:
                Dict with either:
                    - loss and stats (training mode with loss_mask)
                    - logits (inference mode without loss_mask)
            """
            input_ids = kwargs["seqs"]
            conti_feats = kwargs["conti_feats"]
            loss_mask = kwargs.get("loss_masks", None)
            position_ids = kwargs.get("position_ids", None)
            past_key_values = kwargs.get("past_key_values", None)

            inputs_embeds = self._embed(input_ids, conti_feats)

            # Forward through base transformer model
            output = self.model.forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
            )

            # Add stream embeddings to create stream-specific representations
            # Shape: [batch, seq, hidden] -> [batch, seq, streams, hidden]
            hidden_states = output.last_hidden_state.unsqueeze(2)
            stream_emb = self.stream_emb.weight.tile(1, 1, 1, 1)
            stream_emb[:, :, 0] = 0.0  # First stream uses base representation
            hidden_states = hidden_states + stream_emb

            if loss_mask is not None:
                loss, stats = self._loss(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                )
                return {"loss": loss, "stats": stats}

            else:
                logits = self.lm_head(hidden_states)
                return {"logits": logits}

        def _embed(self, input_ids, conti_feats):
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
            for io_name, (io_index, io_feats) in conti_feats.items():
                if not self.multimodal_io_dict[io_name].is_discrete:
                    continue
                # Encode features to discrete codes
                codes = self.multimodal_io_dict[io_name].encode_batch(*io_feats)
                # Add vocabulary offset for this modality
                codes = codes + self.vocab_intervals[io_name][0][0]
                # Place codes in correct positions
                for code, (bidx, start, length) in zip(codes, io_index):
                    input_ids[bidx, start : start + length] = code[:length]

            # (2) Convert tokens to embeddings and sum across streams
            input_embeds = self.model.embed_tokens(input_ids).sum(dim=2)

            # (3) Process continuous modalities: encode and project features
            for io_name, (io_index, io_feats) in conti_feats.items():
                if self.multimodal_io_dict[io_name].is_discrete:
                    continue
                io_feats = self.multimodal_io_dict[io_name].encode_batch(*io_feats)
                for feat, (bidx, start, length) in zip(io_feats, io_index):
                    feat = self.adaptor[io_name](feat)
                    input_embeds[bidx, start : start + length] = feat[:length]

            return input_embeds

        def _loss(self, hidden_states, input_ids, loss_mask):
            """Compute multimodal language modeling loss.

            Uses full vocabulary softmax for first stream (text/special tokens)
            and interval-based softmax for other streams (audio/discrete tokens)
            to efficiently handle large vocabularies.

            Args:
                hidden_states: Model outputs [batch, seq_len, streams, hidden_dim]
                input_ids: Target tokens [batch, seq_len, streams]
                loss_mask: Loss weights per token [batch, seq_len, streams]

            Returns:
                Tuple of (loss tensor, stats dict with loss/accuracy metrics)
            """
            assert input_ids.size() == loss_mask.size()
            assert hidden_states.size()[:3] == loss_mask.size()

            # Shift for next-token prediction
            hidden_states = hidden_states[:, :-1]
            input_ids = input_ids[:, 1:]
            loss_mask = loss_mask[:, 1:]

            # Initialize loss and accuracy tensors
            loss = torch.zeros_like(loss_mask)
            acc = torch.zeros_like(loss_mask).bool()
            stats = dict()

            # Stream 0: Full vocabulary softmax
            this_mask = torch.zeros_like(input_ids).bool()
            this_mask[:, :, 0] = True

            this_logits = hidden_states[this_mask]
            this_logits = torch.matmul(this_logits, self.lm_head.weight.T)
            this_targets = input_ids[this_mask]

            this_loss = torch.nn.functional.cross_entropy(
                this_logits,
                this_targets,
                reduction="none",
                ignore_index=0,
            )
            loss.masked_scatter_(this_mask, this_loss)
            if not self.training:
                this_acc = this_logits.argmax(-1) == this_targets
                acc.masked_scatter_(this_mask, this_acc)

            # Streams 1+: Interval-based softmax for discrete modalities
            # Process each vocabulary interval separately to avoid OOM
            residual_ids = input_ids[:, :, 1:]
            for start, end in self.loss_intervals:
                # Find tokens in this interval
                this_mask = torch.logical_and(residual_ids >= start, residual_ids < end)
                if this_mask.int().sum() == 0:
                    continue
                # Compute loss only for vocabulary subset [start:end]
                this_logits = hidden_states[:, :, 1:][this_mask]
                this_logits = torch.matmul(
                    this_logits, self.lm_head.weight[start:end].T
                )
                # Adjust targets to interval-relative indices
                this_targets = residual_ids[this_mask] - start
                this_loss = torch.nn.functional.cross_entropy(
                    this_logits,
                    this_targets,
                    reduction="none",
                )
                loss[:, :, 1:].masked_scatter_(this_mask, this_loss)
                if not self.training:
                    this_acc = this_logits.argmax(-1) == this_targets
                    acc[:, :, 1:].masked_scatter_(this_mask, this_acc)

            # Apply loss masks and compute weighted average
            loss = loss * loss_mask
            count = (loss_mask != 0.0).float()
            loss = loss.sum() / count.sum()
            stats["loss"] = loss.clone().detach()

            # Compute accuracy statistics during evaluation
            if not self.training:
                acc = acc.float()
                stats["acc"] = acc.sum() / count.sum()  # Overall accuracy
                # Per-stream accuracy for debugging
                for n in range(self.num_stream):
                    this_count = count[:, :, n].sum()
                    if this_count > 0:
                        stats[f"acc_layer{n}"] = acc[:, :, n].sum() / this_count

            return loss, stats

    return ParallelLLM
