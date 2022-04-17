#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.postdecoder.abs_postdecoder import AbsPostDecoder

try:
    from transformers import AutoModel, AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False
from typeguard import check_argument_types
from typing import Tuple

import copy
import logging
import torch


class HuggingFaceTransformersPostDecoder(AbsPostDecoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        model_name_or_path: str,
        output_size=256,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        logging.info("Pretrained Transformers model parameters reloaded!")
        self.out_linear = torch.nn.Linear(self.model.config.hidden_size, output_size)
        self.output_size_dim = output_size
        # if hasattr(model, "encoder"):
        #     self.transformer = model.encoder
        # else:
        #     self.transformer = model

        # if hasattr(self.transformer, "embed_tokens"):
        #     del self.transformer.embed_tokens
        # if hasattr(self.transformer, "wte"):
        #     del self.transformer.wte
        # if hasattr(self.transformer, "word_embedding"):
        #     del self.transformer.word_embedding

        # self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        # if (
        #     self.transformer.config.is_encoder_decoder
        #     or self.transformer.config.model_type in ["xlnet", "t5"]
        # ):
        #     self.use_inputs_embeds = True
        #     self.extend_attention_mask = False
        # elif self.transformer.config.model_type == "gpt2":
        #     self.use_inputs_embeds = True
        #     self.extend_attention_mask = True
        # else:
        #     self.use_inputs_embeds = False
        #     self.extend_attention_mask = True

        # self.linear_in = torch.nn.Linear(
        #     input_size, self.transformer.config.hidden_size
        # )

    def forward(
        self,
        transcript_input_ids: torch.LongTensor,
        transcript_attention_mask: torch.LongTensor,
        transcript_token_type_ids: torch.LongTensor,
        transcript_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward."""
        # input = self.linear_in(input)

        # args = {"return_dict": True}

        # mask = (~make_pad_mask(input_lengths)).to(input.device).float()

        # if self.extend_attention_mask:
        #     args["attention_mask"] = _extend_attention_mask(mask)
        # else:
        #     args["attention_mask"] = mask

        # if self.use_inputs_embeds:
        #     args["inputs_embeds"] = input
        # else:
        #     args["hidden_states"] = input

        # if self.transformer.config.model_type == "mpnet":
        #     args["head_mask"] = [None for _ in self.transformer.layer]

        # output = self.transformer(**args).last_hidden_state
        # print(transcript_input_ids)
        # print(transcript_input_ids.size())
        # print(transcript_attention_mask)
        # print(transcript_attention_mask.shape)
        # print(transcript_token_type_ids)
        # print(transcript_token_type_ids.shape)
        # print(self.model.config)
        # print(self.model.embeddings.position_embeddings.weight.data.shape)
        transcript_outputs = self.model(
            input_ids=transcript_input_ids,
            position_ids=transcript_position_ids,
            attention_mask=transcript_attention_mask,
            token_type_ids=transcript_token_type_ids,
        )

        return self.out_linear(transcript_outputs.last_hidden_state)
        # return self.out_linear(transcript_outputs.pooler_output)

    # def reload_pretrained_parameters(self):
    #     self.transformer.load_state_dict(self.pretrained_params)
    #     logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_size_dim

    def convert_examples_to_features(self, data, max_seq_length):
        input_id_features = []
        input_mask_features = []
        segment_ids_feature = []
        position_ids_feature = []
        input_id_length = []
        for text_id in range(len(data)):
            tokens_a = self.tokenizer.tokenize(data[text_id])
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_id_length.append(len(input_ids))
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            position_ids = [i for i in range(max_seq_length)]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(position_ids) == max_seq_length
            input_id_features.append(input_ids)
            input_mask_features.append(input_mask)
            segment_ids_feature.append(segment_ids)
            position_ids_feature.append(position_ids)
        return (
            input_id_features,
            input_mask_features,
            segment_ids_feature,
            position_ids_feature,
            input_id_length,
        )


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
