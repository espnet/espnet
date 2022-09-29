from abc import ABC, abstractmethod

import torch


class AbsPostDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        transcript_input_ids: torch.LongTensor,
        transcript_attention_mask: torch.LongTensor,
        transcript_token_type_ids: torch.LongTensor,
        transcript_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def convert_examples_to_features(
        self, data: list, max_seq_length: int, output_size: int
    ):
        raise NotImplementedError
