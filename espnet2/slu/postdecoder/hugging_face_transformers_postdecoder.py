#!/usr/bin/env python3
#  2022, Carnegie Mellon University;  Siddhant Arora
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostDecoder."""

from espnet2.slu.postdecoder.abs_postdecoder import AbsPostDecoder

try:
    from transformers import AutoModel, AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False
import logging

import torch
from typeguard import typechecked


class HuggingFaceTransformersPostDecoder(AbsPostDecoder):
    """
    Hugging Face Transformers PostDecoder.

    This class is responsible for decoding outputs from a pretrained 
    Hugging Face Transformers model. It utilizes the transformers 
    library to load models and tokenizers, and processes input 
    sequences for downstream tasks in spoken language understanding 
    (SLU).

    Attributes:
        model (transformers.AutoModel): The loaded Hugging Face model.
        tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
        out_linear (torch.nn.Linear): Linear layer for output transformation.
        output_size_dim (int): Dimension of the output size.

    Args:
        model_name_or_path (str): The model name or path to the pretrained model.
        output_size (int, optional): The size of the output layer. Defaults to 256.

    Raises:
        ImportError: If the `transformers` library is not installed.

    Examples:
        >>> post_decoder = HuggingFaceTransformersPostDecoder(
        ...     model_name_or_path='bert-base-uncased',
        ...     output_size=128
        ... )
        >>> input_ids, attention_mask, token_type_ids, position_ids = post_decoder.convert_examples_to_features(
        ...     ["Hello, how are you?"], max_seq_length=20
        ... )
        >>> outputs = post_decoder.forward(input_ids[0], attention_mask[0], 
        ...                                  token_type_ids[0], position_ids[0])
        >>> print(outputs.shape)
        torch.Size([20, 128])
    """

    @typechecked
    def __init__(
        self,
        model_name_or_path: str,
        output_size=256,
    ):
        """Initialize the module."""
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

    def forward(
        self,
        transcript_input_ids: torch.LongTensor,
        transcript_attention_mask: torch.LongTensor,
        transcript_token_type_ids: torch.LongTensor,
        transcript_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        This method takes input tensors for the model and processes them
        through the Hugging Face Transformers model, followed by a linear
        transformation to produce the final output.

        Args:
            transcript_input_ids (torch.LongTensor): The input token IDs 
                for the transcripts.
            transcript_attention_mask (torch.LongTensor): The attention mask 
                to avoid attending to padding tokens.
            transcript_token_type_ids (torch.LongTensor): Token type IDs 
                to distinguish between different segments.
            transcript_position_ids (torch.LongTensor): Position IDs to 
                indicate the position of tokens in the input sequence.

        Returns:
            torch.Tensor: The output tensor after applying the linear 
            transformation to the model's last hidden state.

        Examples:
            >>> model = HuggingFaceTransformersPostDecoder("bert-base-uncased")
            >>> input_ids = torch.tensor([[101, 2023, 2003, 1037, 3391, 102]])
            >>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
            >>> token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
            >>> position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
            >>> output = model.forward(input_ids, attention_mask, token_type_ids, 
            ...                        position_ids)
            >>> print(output.shape)  # Should output the shape of the transformed tensor.

        Raises:
            ValueError: If the input tensors are not of compatible shapes.
        """
        transcript_outputs = self.model(
            input_ids=transcript_input_ids,
            position_ids=transcript_position_ids,
            attention_mask=transcript_attention_mask,
            token_type_ids=transcript_token_type_ids,
        )

        return self.out_linear(transcript_outputs.last_hidden_state)

    def output_size(self) -> int:
        """
        Get the output size of the post-decoder.

        This method retrieves the size of the output layer of the 
        Hugging Face Transformers PostDecoder, which is defined during 
        initialization. The output size is typically used to determine 
        the dimensionality of the output tensor produced by the forward 
        pass of the model.

        Returns:
            int: The output size specified during the initialization of 
            the HuggingFaceTransformersPostDecoder.

        Examples:
            >>> decoder = HuggingFaceTransformersPostDecoder("bert-base-uncased", output_size=128)
            >>> decoder.output_size()
            128

        Note:
            The output size can be adjusted according to the requirements 
            of the downstream task, such as classification or regression.
        """
        return self.output_size_dim

    def convert_examples_to_features(self, data, max_seq_length):
        """
        Converts input text examples into features for model processing.

        This method tokenizes input text examples and converts them into a format
        suitable for input into a transformer model. The output includes input IDs,
        attention masks, segment IDs, position IDs, and lengths of the input IDs.

        Args:
            data (List[str]): A list of input text examples to be tokenized.
            max_seq_length (int): The maximum sequence length for the tokenized
                inputs. Sequences longer than this will be truncated.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]], 
                  List[List[int]], List[int]]:
                - A list of input IDs for each example.
                - A list of attention masks for each example.
                - A list of segment IDs for each example.
                - A list of position IDs for each example.
                - A list containing the lengths of the input ID sequences.

        Raises:
            AssertionError: If the lengths of any of the generated features do not
                match `max_seq_length`.

        Examples:
            >>> decoder = HuggingFaceTransformersPostDecoder("bert-base-uncased")
            >>> data = ["Hello, world!", "This is a test."]
            >>> features = decoder.convert_examples_to_features(data, max_seq_length=10)
            >>> len(features)
            (2, 2, 2, 2, 2)  # Corresponds to input_ids, attention_mask, segment_ids,
                              # position_ids, and input_id_length respectively.

        Note:
            The method prepends the "[CLS]" token and appends the "[SEP]" token
            to each example. Padding is applied to ensure that all sequences have
            the same length as specified by `max_seq_length`.
        """
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
