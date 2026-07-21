#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

import copy
import logging
from typing import Dict, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoConfig, AutoModel, AutoModelForPreTraining

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersEncoder(AbsEncoder):
    """Hugging Face Transformers PostEncoder.

    By default this wraps a (text) Transformer and returns its
    ``last_hidden_state`` as ``(output, output_lengths)`` -- e.g. as a
    post-encoder in the ST/ASR tasks.

    When ``do_pretrain=True`` it instead wraps a self-supervised speech model
    such as WavLM (loaded via ``AutoModelForPreTraining``) and returns a dict
    containing the model's self-supervised ``"loss"``. In this mode the encoder
    consumes raw audio, masks time steps itself, and lets the model predict the
    k-means cluster id of each masked frame (HuBERT-style masked prediction), so
    it can be driven directly by ``espnet2.wavlm.espnet_model.WavLMPretrainModel``.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
        lang_token_id: int = -1,
        do_pretrain: bool = False,
        num_classes: Optional[int] = None,
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_min_masks: int = 2,
        final_dim: int = 256,
        logit_temp: float = 0.1,
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
    ):
        """Initialize the module.

        Args:
            input_size: Input feature dimension (kept for the ``AbsEncoder``
                interface; unused by the Transformers model itself).
            model_name_or_path: Hugging Face model id or local path.
            lang_token_id: Optional language token prepended to the input in
                the (text) post-encoder mode.
            do_pretrain: If True, load a self-supervised speech model for
                WavLM-style masked cluster prediction and return the loss from
                ``forward``.
            num_classes: Number of k-means clusters (the pretraining target
                vocabulary). Required when ``do_pretrain=True``.
            mask_prob: Probability that a frame is the start of a masked span
                (pretraining only).
            mask_length: Length of each masked span (pretraining only).
            mask_min_masks: Minimum number of masked spans per sample
                (pretraining only).
            final_dim: Projection dim used to score frames against cluster
                embeddings (pretraining only).
            logit_temp: Temperature for the cosine-similarity logits
                (pretraining only).
            pred_masked_weight: Weight on the masked-frame prediction loss
                (pretraining only).
            pred_nomask_weight: Weight on the unmasked-frame prediction loss
                (pretraining only).
        """
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.do_pretrain = do_pretrain

        if do_pretrain:
            if num_classes is None:
                raise ValueError(
                    "`num_classes` (number of k-means clusters) is required when "
                    "`do_pretrain=True`."
                )

            # Inject the pretraining-head hyper-parameters into the config so
            # that WavLMForPreTraining (built by from_pretrained) can read them.
            config = AutoConfig.from_pretrained(model_name_or_path)
            config.num_classes = num_classes
            config.wavlm_final_dim = final_dim
            config.wavlm_logit_temp = logit_temp
            config.wavlm_pred_masked_weight = pred_masked_weight
            config.wavlm_pred_nomask_weight = pred_nomask_weight

            # Self-supervised speech model (e.g. WavLM). Loads the pretrained
            # backbone; the prediction head is newly initialized.
            self.transformer = AutoModelForPreTraining.from_pretrained(
                model_name_or_path, config=config
            )
            self.mask_prob = mask_prob
            self.mask_length = mask_length
            self.mask_min_masks = mask_min_masks
        else:
            model = AutoModel.from_pretrained(model_name_or_path)

            if hasattr(model, "encoder"):
                self.transformer = model.encoder
            else:
                self.transformer = model

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        self.lang_token_id = lang_token_id

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        ys_pad: Optional[torch.Tensor] = None,
        ys_pad_lengths: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward.

        In the default (post-encoder) mode returns
        ``(last_hidden_state, input_lengths)`` and ignores ``ys_pad``. In
        pretraining mode returns a dict with at least a ``"loss"`` key, using
        ``ys_pad`` as the per-frame k-means cluster targets.
        """
        if self.do_pretrain:
            return self._pretrain_forward(input, input_lengths, ys_pad)

        args = {"return_dict": True}

        if self.lang_token_id != -1:
            input = torch.cat(
                (
                    torch.tensor(
                        [self.lang_token_id] * input.shape[0], device=input.device
                    ).unsqueeze(1),
                    input,
                ),
                dim=-1,
            )
            input_lengths = input_lengths + 1

        args["input_ids"] = input

        mask = (~make_pad_mask(input_lengths)).to(input.device).float()
        args["attention_mask"] = mask
        output = self.transformer(**args).last_hidden_state

        return output, input_lengths

    def _pretrain_forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        ys_pad: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """WavLM-style masked cluster-prediction forward returning the loss.

        Args:
            input: Raw audio waveform, ``(Batch, NSamples)``.
            input_lengths: Number of valid samples per utterance, ``(Batch,)``.
            ys_pad: Per-frame k-means cluster ids, ``(Batch, Frames)``.
        """
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            _compute_mask_indices,
        )

        batch_size = input.shape[0]

        # Sample-level padding mask (1 = keep) so the Transformer ignores
        # padded audio.
        attention_mask = (~make_pad_mask(input_lengths)).to(input.device).long()

        # Number of feature-extractor output frames after conv downsampling.
        output_lengths = self.transformer._get_feat_extract_output_lengths(
            input_lengths
        )
        seq_len = int(output_lengths.max())

        # Choose which frames to mask (span masking, wav2vec2/WavLM defaults).
        mask_time_indices_np = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
            min_masks=self.mask_min_masks,
        )
        mask_time_indices = torch.from_numpy(mask_time_indices_np).to(input.device)

        outputs = self.transformer(
            input_values=input,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            target=ys_pad,
            return_dict=True,
        )

        return outputs

    def reload_pretrained_parameters(self):
        self.transformer.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """Get the output size."""
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
