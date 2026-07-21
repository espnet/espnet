#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# WavLM self-supervised pretraining head.
#     Paper: https://arxiv.org/abs/2110.13900
#
# Hugging Face ships ``WavLMModel`` but no ``WavLMForPreTraining`` head. This
# module provides one that follows WavLM's (HuBERT-style) objective: predict the
# discrete k-means cluster id of each *masked* frame with a cross-entropy loss.
# The cluster targets are the ones the ``wavlm1`` recipe already produces
# (stage 5 k-means -> ``text.km`` -> the ``text`` field, with ``--num_classes``
# set to the number of clusters).
#
# It is registered with ``AutoModelForPreTraining`` so that
# ``HuggingFaceTransformersEncoder(do_pretrain=True)`` can load it via
# ``AutoModelForPreTraining.from_pretrained("microsoft/wavlm-*", config=...)``.
#
# NOTE: This implements the masked-prediction objective. WavLM's other defining
# ingredient -- utterance mixing / speech denoising -- lives on the data path
# and is NOT added here.

import logging

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoModelForPreTraining
    from transformers.models.wavlm.configuration_wavlm import WavLMConfig
    from transformers.models.wavlm.modeling_wavlm import (
        WavLMModel,
        WavLMPreTrainedModel,
    )

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


if is_transformers_available:

    class WavLMForPreTraining(WavLMPreTrainedModel):
        """WavLM with a HuBERT-style masked cluster-prediction head.

        The number of clusters and the loss hyper-parameters are read from the
        ``config`` (injected by ``HuggingFaceTransformersEncoder`` before
        loading), so they can be driven from the recipe / ``encoder_conf``:

            config.num_classes             -- number of k-means clusters (K)
            config.wavlm_final_dim         -- projection dim for the logits
            config.wavlm_logit_temp        -- temperature for cosine logits
            config.wavlm_pred_masked_weight-- weight on masked-frame loss
            config.wavlm_pred_nomask_weight-- weight on unmasked-frame loss
            config.wavlm_ignore_id         -- label pad id to ignore (default -1)
        """

        config_class = WavLMConfig
        base_model_prefix = "wavlm"
        main_input_name = "input_values"

        def __init__(self, config: "WavLMConfig"):
            super().__init__(config)
            self.wavlm = WavLMModel(config)

            self.num_classes = getattr(config, "num_classes", None)
            self.final_dim = getattr(config, "wavlm_final_dim", 256)
            self.logit_temp = getattr(config, "wavlm_logit_temp", 0.1)
            self.pred_masked_weight = getattr(config, "wavlm_pred_masked_weight", 1.0)
            self.pred_nomask_weight = getattr(config, "wavlm_pred_nomask_weight", 0.0)
            self.ignore_id = getattr(config, "wavlm_ignore_id", -1)

            # Prediction head: project the encoder output, then score it against
            # a learned embedding per cluster via (temperature-scaled) cosine
            # similarity -- as in fairseq HuBERT's final layer.
            if self.num_classes is not None:
                self.final_proj = nn.Linear(config.hidden_size, self.final_dim)
                self.label_embeddings = nn.Embedding(self.num_classes, self.final_dim)
            else:
                self.final_proj = None
                self.label_embeddings = None

            # Initialize weights and apply final processing
            self.post_init()

        def freeze_feature_encoder(self):
            """Disable gradient updates for the feature encoder."""
            self.wavlm.feature_extractor._freeze_parameters()

        def _logits(self, hidden_states):
            """Cosine-similarity logits over the cluster embeddings.

            Args:
                hidden_states: (Batch, Time, Hidden)
            Returns:
                (Batch, Time, num_classes)
            """
            projected = F.normalize(self.final_proj(hidden_states), dim=-1)
            embeddings = F.normalize(self.label_embeddings.weight, dim=-1)
            logits = torch.matmul(projected, embeddings.transpose(0, 1))
            return logits / self.logit_temp

        def forward(
            self,
            input_values,
            attention_mask=None,
            mask_time_indices=None,
            target=None,
            target_lengths=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            if self.label_embeddings is None:
                raise RuntimeError(
                    "WavLMForPreTraining needs `num_classes` (number of k-means "
                    "clusters) set on the config to build its prediction head. "
                    "Pass it via the encoder's `num_classes` argument."
                )

            if mask_time_indices is not None:
                mask_time_indices = mask_time_indices.to(torch.bool)

            # WavLMModel masks the given time steps with its learned mask
            # embedding before the Transformer layers.
            outputs = self.wavlm(
                input_values,
                attention_mask=attention_mask,
                mask_time_indices=mask_time_indices,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            hidden_states = outputs[0]  # (B, T, H)

            logits = self._logits(hidden_states)  # (B, T, K)

            if target is None:
                # No targets (e.g. feature extraction): return representations.
                return {"logits": logits, "hidden_states": hidden_states}

            # Align frame-level predictions and cluster targets. The collate
            # function downsamples labels to roughly the feature frame rate, but
            # off-by-a-few is possible; truncate to the common length.
            length = min(logits.size(1), target.size(1))
            logits = logits[:, :length]
            target = target[:, :length].long()

            if mask_time_indices is not None:
                mask = mask_time_indices[:, :length]
            else:
                mask = torch.zeros_like(target, dtype=torch.bool)

            num_classes = logits.size(-1)
            logits_flat = logits.reshape(-1, num_classes)
            target_flat = target.reshape(-1)
            mask_flat = mask.reshape(-1)

            valid = target_flat != self.ignore_id
            masked_sel = mask_flat & valid
            nomask_sel = (~mask_flat) & valid

            loss = logits_flat.new_zeros(())

            if self.pred_masked_weight > 0 and masked_sel.any():
                loss_masked = F.cross_entropy(
                    logits_flat[masked_sel],
                    target_flat[masked_sel],
                    reduction="sum",
                )
                loss = loss + self.pred_masked_weight * loss_masked

            if self.pred_nomask_weight > 0 and nomask_sel.any():
                loss_nomask = F.cross_entropy(
                    logits_flat[nomask_sel],
                    target_flat[nomask_sel],
                    reduction="sum",
                )
                loss = loss + self.pred_nomask_weight * loss_nomask

            with torch.no_grad():
                acc_mask = self._accuracy(logits_flat, target_flat, masked_sel)
                acc_unmask = self._accuracy(logits_flat, target_flat, nomask_sel)

            return {
                "loss": loss,
                "acc_mask": acc_mask,
                "acc_unmask": acc_unmask,
            }

        @staticmethod
        def _accuracy(logits_flat, target_flat, selection):
            if selection.any():
                pred = logits_flat[selection].argmax(dim=-1)
                correct = (pred == target_flat[selection]).float().sum()
                return correct / selection.long().sum()
            return logits_flat.new_zeros(())

    # Make ``AutoModelForPreTraining.from_pretrained`` return WavLMForPreTraining
    # for WavLM checkpoints. Guard against double registration.
    try:
        AutoModelForPreTraining.register(WavLMConfig, WavLMForPreTraining)
    except Exception as e:  # already registered or unsupported transformers ver
        logging.debug(f"Could not register WavLMForPreTraining: {e}")

else:

    WavLMForPreTraining = None
