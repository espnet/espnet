#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

import copy
import logging
import os
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.asr.asr_utils import get_model_conf
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

from espnet.nets.scorer_interface import BatchScorerInterface


class HuggingFaceTransformersDecoder(AbsDecoder, BatchScorerInterface):
    """
    Hugging Face Transformers Decoder.

    This class implements a decoder that utilizes Hugging Face's Transformers
    models for automatic speech recognition (ASR). It supports both causal
    language models and sequence-to-sequence models.

    Args:
        vocab_size (int): The size of the vocabulary.
        encoder_output_size (int): The size of the encoder output.
        model_name_or_path (str): The name or path of the pre-trained
            Transformers model.
        causal_lm (bool, optional): Whether to use a causal language model.
            Defaults to False. This overrides the model_name_or_path if
            provided.
        prefix (str, optional): Prefix to be added to the input tokens.
            Defaults to "".
        postfix (str, optional): Postfix to be added to the input tokens.
            Defaults to "".
        overriding_architecture_config (str or dict, optional): Path to the
            configuration JSON file or the JSON dictionary itself. Defaults
            to None. If this is set, it can be used to override the default
            decoder configuration.
        load_pretrained_weights (bool): Whether to load the pre-trained
            weights. Defaults to True.
        separate_lm_head (bool): True ensures that the language model head
            is not shared with the input token embeddings. When False, the
            original structure is kept, i.e., if the original Transformers
            implementation has tying of weights, it is retained. Defaults
            to False.

    Raises:
        ImportError: If the `transformers` library is not available.
        Exception: If the word embeddings attribute cannot be found in
            the model.

    Examples:
        >>> decoder = HuggingFaceTransformersDecoder(
        ...     vocab_size=5000,
        ...     encoder_output_size=256,
        ...     model_name_or_path="gpt2",
        ...     causal_lm=True
        ... )
        >>> hs_pad = torch.rand(32, 10, 256)  # Example encoder output
        >>> hlens = torch.tensor([10] * 32)  # Example lengths
        >>> ys_in_pad = torch.randint(0, 5000, (32, 15))  # Example input
        >>> ys_in_lens = torch.tensor([15] * 32)  # Example lengths
        >>> output, output_lengths = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        Ensure that the `transformers` library is installed to use this class.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
        causal_lm: bool = False,
        prefix: str = "",
        postfix: str = "",
        overriding_architecture_config: Optional[Union[str, dict]] = {},
        load_pretrained_weights: bool = True,
        separate_lm_head: bool = False,
    ):
        """
        Initializes the HuggingFaceTransformersDecoder.

        Args:
            vocab_size (int): The size of the vocabulary.
            encoder_output_size (int): The size of the encoder output.
            model_name_or_path (str): The name or path of the pre-trained
                 Transformers model.
            causal_lm (bool, optional): Whether to use a causal language
                model. Defaults to False. This overrides the
                model_name_or_path if provided.
            prefix (str, optional): Prefix to be added to the input
                tokens. Defaults to "".
            postfix (str, optional): Postfix to be added to the input
                tokens. Defaults to "".
            overriding_architecture_config (str or dict, optional): Path to the
                configuration json file or the json dictionary itself. Defaults
                to None. If this is set, it can be used to override the
                default decoder configuration.
            load_pretrained_weights (bool): Whether to load the pre-trained
                weights. Defaults to True.
            separate_lm_head (bool): True ensures that the language model
                head is not shared with the input token embeddings. When False,
                the original structure is kept, ie, if the original Transformers
                implementation has tying of weights, it is retained. Defaults
                to False.

        Raises:
            ImportError: If the `transformers` library is not available.
            Exception: If the word embeddings attribute cannot be found in
                the model.
        """
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it "
                "via `pip install"
                " transformers` or `cd /path/to/espnet/tools "
                "&& . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.load_pretrained_weights = load_pretrained_weights
        self.separate_lm_head = separate_lm_head

        self.overriding_architecture_config = overriding_architecture_config
        if isinstance(overriding_architecture_config, str):
            # It is path to a json config file
            self.overriding_architecture_config = vars(
                get_model_conf(model_path="", conf_path=overriding_architecture_config)
            )

        self.causal_lm = causal_lm

        if self.causal_lm:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **self.overriding_architecture_config
            )
            self.decoder = get_hugging_face_model_network(model)

            if hasattr(self.decoder, "word_embeddings"):
                self.decoder_word_embeddings = self.decoder.word_embeddings
            elif hasattr(self.decoder, "embed_in"):
                self.decoder_word_embeddings = self.decoder.embed_in
            elif hasattr(self.decoder, "embed_tokens"):
                self.decoder_word_embeddings = self.decoder.embed_tokens
            else:
                raise Exception("Can not find the word embeddings attribute")

            if (
                self.decoder.config.pad_token_id is not None
                and self.decoder.config.pad_token_id != -1
            ):
                self.decoder_pad_token_id = self.decoder.config.pad_token_id
            else:
                self.decoder_pad_token_id = 1

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.tokenizer_padding_side = tokenizer.padding_side

            self.prefix = self.decoder_word_embeddings(
                tokenizer.encode(prefix, return_tensors="pt").long()
            ).detach()

            self.postfix = self.decoder_word_embeddings(
                tokenizer.encode(postfix, return_tensors="pt").long()
            ).detach()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, **self.overriding_architecture_config
            )

            if hasattr(model, "model"):
                self.decoder = model.model.decoder
            else:
                self.decoder = model.decoder

        model.resize_token_embeddings(vocab_size)

        if self.separate_lm_head:
            self.lm_head = copy.deepcopy(get_hugging_face_model_lm_head(model))
        else:
            self.lm_head = get_hugging_face_model_lm_head(model)

        self.model_name_or_path = model_name_or_path

        self.decoder_pretrained_params = copy.deepcopy(self.decoder.state_dict())
        self.lm_head_pretrained_params = copy.deepcopy(self.lm_head.state_dict())

        if encoder_output_size != self.decoder.config.hidden_size:
            self.linear_in = torch.nn.Linear(
                encoder_output_size, self.decoder.config.hidden_size
            )
        else:
            self.linear_in = torch.nn.Identity()

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the decoder.

        This method processes the encoded memory from the encoder and the
        input tensor to generate token scores before softmax. It can handle
        both causal language models and sequence-to-sequence models based on
        the initialization parameters.

        Args:
            hs_pad (torch.Tensor): Encoded memory from the encoder with shape
                (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of the encoded sequences with shape
                (batch).
            ys_in_pad (torch.Tensor): Input tensor for the decoder with shape
                (batch, maxlen_out, #mels).
            ys_in_lens (torch.Tensor): Lengths of the input sequences with
                shape (batch).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax with
                  shape (batch, maxlen_out, token).
                - olens (torch.Tensor): Lengths of the output sequences with
                  shape (batch,).

        Examples:
            >>> decoder = HuggingFaceTransformersDecoder(...)
            >>> hs_pad = torch.rand(2, 10, 512)  # Example encoded memory
            >>> hlens = torch.tensor([10, 8])  # Example lengths
            >>> ys_in_pad = torch.rand(2, 5, 80)  # Example input tensor
            >>> ys_in_lens = torch.tensor([5, 4])  # Example lengths
            >>> scores, lengths = decoder.forward(hs_pad, hlens, ys_in_pad,
            ...                                     ys_in_lens)

        Note:
            This method assumes that the model has been initialized with
            appropriate parameters, including whether it is a causal language
            model or a sequence-to-sequence model.

        Raises:
            ValueError: If the shapes of the input tensors do not match the
            expected dimensions.
        """
        enc_out = self.linear_in(hs_pad)

        if self.causal_lm:
            args, no_loss_lengths = self.add_prefix_postfix(
                enc_out, hlens, ys_in_pad, ys_in_lens
            )
        else:
            args = {"return_dict": True}

            if self.decoder.__class__.__name__ == "MBartDecoder":
                ys_in_pad[:, 0] = 2

            args["input_ids"] = ys_in_pad
            mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
            args["attention_mask"] = mask

            args["encoder_hidden_states"] = enc_out
            hs_mask = (~make_pad_mask(hlens)).to(hs_pad.device).float()
            args["encoder_attention_mask"] = hs_mask

        x = self.decoder(**args).last_hidden_state

        if self.causal_lm:
            if self.tokenizer_padding_side == "left":
                x = torch.vstack(
                    [
                        F.pad(
                            x[i, -ys_in_lens[i] :, :],
                            (0, 0, 0, ys_in_lens.max() - ys_in_lens[i]),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )
            else:
                x = torch.vstack(
                    [
                        F.pad(
                            x[
                                i,
                                no_loss_lengths[i] : no_loss_lengths[i] + ys_in_lens[i],
                                :,
                            ],
                            (0, 0, 0, ys_in_lens.max() - ys_in_lens[i]),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )

        x = self.lm_head(x)

        return x, ys_in_lens

    def reload_pretrained_parameters(self):
        """
        Reloads the pretrained parameters for the decoder and language model head.

        This method is designed to load the previously saved pretrained parameters
        for the decoder and its language model head if the `load_pretrained_weights`
        attribute is set to True. If loading is skipped, a corresponding log message
        is generated.

        Attributes:
            load_pretrained_weights (bool): Indicates whether to load pretrained
                weights or not.

        Raises:
            Exception: If there are issues loading the pretrained parameters.

        Examples:
            >>> decoder = HuggingFaceTransformersDecoder(
            ...     vocab_size=1000,
            ...     encoder_output_size=512,
            ...     model_name_or_path='gpt2',
            ... )
            >>> decoder.reload_pretrained_parameters()
            Loaded pretrained Transformers decoder parameters!
        """
        if self.load_pretrained_weights:
            self.decoder.load_state_dict(self.decoder_pretrained_params)
            logging.info("Loaded pretrained Transformers decoder parameters!")

            if self.lm_head_pretrained_params is not None:
                self.lm_head.load_state_dict(self.lm_head_pretrained_params)
                logging.info("Loaded pretrained Transformers LM head parameters!")
        else:
            logging.info(
                "Skipping the loading of pretrained Transformer model parameters!"
            )

    def add_prefix_postfix(self, enc_out, hlens, ys_in_pad, ys_in_lens):
        """
        Adds a prefix and postfix to the encoder output for token input during decoding.

        This method constructs the input for the decoder by concatenating a prefix,
        the encoder output, a postfix, and the input token embeddings. It also
        generates the appropriate attention mask for the decoder.

        Args:
            enc_out (torch.Tensor): The encoded output from the encoder. Shape
                should be (batch_size, max_length, hidden_size).
            hlens (torch.Tensor): A tensor containing the lengths of the encoder
                outputs for each sample in the batch. Shape should be (batch_size,).
            ys_in_pad (torch.Tensor): Input tensor representing the target
                sequence. Shape should be (batch_size, max_length_out).
            ys_in_lens (torch.Tensor): A tensor containing the lengths of the
                input target sequences. Shape should be (batch_size,).

        Returns:
            Tuple[dict, torch.Tensor]: A tuple containing:
                - args (dict): A dictionary of inputs prepared for the decoder,
                    including 'inputs_embeds' and 'attention_mask'.
                - no_loss_lengths (torch.Tensor): A tensor containing the lengths
                    of the input sequences that will not contribute to the loss
                    calculation.

        Examples:
            >>> prefix = "Hello"
            >>> postfix = "World"
            >>> enc_out = torch.rand(2, 10, 768)  # Example encoder output
            >>> hlens = torch.tensor([10, 8])
            >>> ys_in_pad = torch.tensor([[1, 2, 3], [1, 2, 0]])
            >>> ys_in_lens = torch.tensor([3, 2])
            >>> args, no_loss_lengths = decoder.add_prefix_postfix(enc_out, hlens,
            ...                                                  ys_in_pad, ys_in_lens)

        Note:
            The method handles padding on either the left or right side based on
            the tokenizer's padding configuration. Ensure that the tokenizer
            is correctly initialized before calling this method.
        """
        args = {}

        hlens_max = (hlens + ys_in_lens).max()

        enc_out_list = []

        for i in range(len(hlens)):
            enc_out_element = [
                self.prefix.to(enc_out.device),
                enc_out[i : i + 1, : hlens[i], :],
                self.postfix.to(enc_out.device),
                self.decoder_word_embeddings(
                    ys_in_pad[i : i + 1, 1 : ys_in_lens[i]]
                ).to(enc_out.device),
            ]

            padding = self.decoder_word_embeddings(
                torch.tensor([[self.decoder_pad_token_id]]).to(enc_out.device)
            ).expand(-1, hlens_max - (hlens[i] + ys_in_lens[i]), -1)

            if self.tokenizer_padding_side == "left":
                enc_out_element.insert(0, padding)
            else:
                enc_out_element.insert(len(enc_out_element), padding)

            enc_out_list.append(torch.cat(enc_out_element, dim=1))

        args["inputs_embeds"] = torch.vstack(enc_out_list)

        no_loss_lengths = self.prefix.size(1) + hlens + self.postfix.size(1) - 1
        inputs_lengths = no_loss_lengths + ys_in_lens

        hs_mask = (~make_pad_mask(inputs_lengths)).to(enc_out.device).float()

        if self.tokenizer_padding_side == "left":
            args["attention_mask"] = hs_mask.flip([1])
        else:
            args["attention_mask"] = hs_mask

        args["return_dict"] = True

        return args, no_loss_lengths

    def score(self, ys, state, x, speech=None):
        """
        Scores the next token in a sequence given the current input.

        This method computes the score for the next token based on the current
        state of the decoder and the input sequence. It utilizes the Hugging Face
        Transformers framework to perform the necessary computations.

        Args:
            ys (torch.Tensor): The input tensor representing the sequence of
                tokens (batch_size, sequence_length).
            state (Any): The current state of the decoder, which may contain
                necessary context for scoring.
            x (torch.Tensor): The encoder outputs from the previous step
                (batch_size, encoder_output_size).
            speech (torch.Tensor, optional): Optional input tensor representing
                the speech features, if applicable. Defaults to None.

        Returns:
            Tuple[torch.Tensor, None]: A tuple containing:
                - next_token_scores (torch.Tensor): Log probabilities of the
                  next token (batch_size * num_beams, vocab_size).
                - None: Placeholder for future extension (currently unused).

        Examples:
            >>> decoder = HuggingFaceTransformersDecoder(...)
            >>> ys = torch.tensor([[1, 2, 3]])  # Example input tensor
            >>> state = ...  # Some decoder state
            >>> x = torch.rand(1, encoder_output_size)  # Encoder output
            >>> scores, _ = decoder.score(ys, state, x)
            >>> print(scores.shape)  # (1, vocab_size)

        Note:
            This method currently does not implement caching, which could
            improve performance for successive calls.
        """
        model_kwargs = {
            "encoder_outputs": ModelOutput(
                last_hidden_state=self.linear_in(x).unsqueeze(0)
            ),
        }
        # TODO(brian): caching
        model_inputs = self.hf_generate.prepare_inputs_for_generation(
            ys.unsqueeze(0), **model_kwargs
        )
        outputs = self.hf_generate(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = torch.nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        return next_token_scores.squeeze(0), None

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        speech: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
        Computes the batch scores for a sequence of input tokens.

        This method processes the input sequences and calculates the scores
        for the next token predictions based on the encoder outputs.

        Args:
            ys (torch.Tensor): Tensor of shape (batch_size, sequence_length)
                containing the input sequences for which scores are to be
                computed.
            states (List[Any]): A list of states, which can be used to
                maintain information across the decoding steps.
            xs (torch.Tensor): Tensor of shape (batch_size, feature_size)
                representing the encoder outputs for the corresponding
                sequences.
            speech (torch.Tensor, optional): Optional tensor representing
                speech inputs. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[Any]]: A tuple containing:
                - next_token_scores (torch.Tensor): Tensor of shape
                  (batch_size, vocab_size) containing the log probabilities
                  of the next tokens.
                - List[Any]: The updated list of states after processing
                  the input sequences.

        Examples:
            >>> decoder = HuggingFaceTransformersDecoder(...)
            >>> ys = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> states = [None, None]
            >>> xs = torch.randn(2, 256)  # Example encoder outputs
            >>> scores, new_states = decoder.batch_score(ys, states, xs)
            >>> print(scores.shape)  # Should print: torch.Size([2, vocab_size])

        Note:
            Ensure that the input tensors are properly padded and formatted
            before passing them to this method.

        Raises:
            ValueError: If the input tensors have mismatched dimensions or
                are not compatible with the model.
        """
        # import pdb;pdb.set_trace()
        model_kwargs = {
            "encoder_outputs": ModelOutput(last_hidden_state=self.linear_in(xs)),
        }
        model_inputs = self.hf_generate.prepare_inputs_for_generation(
            ys, **model_kwargs
        )
        outputs = self.hf_generate(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = torch.nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        return next_token_scores, None


def get_hugging_face_model_network(model):
    """
    Retrieve the network from a Hugging Face model.

    This function checks for the presence of specific attributes in the
    provided model and returns the corresponding network. It is designed
    to support various model architectures available in the Hugging Face
    Transformers library.

    Args:
        model: A Hugging Face model instance from which the network is to
            be retrieved.

    Returns:
        The network component of the Hugging Face model.

    Raises:
        Exception: If the model does not contain a recognizable network
            attribute.

    Examples:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> network = get_hugging_face_model_network(model)
        >>> print(network)  # Should print the network component of the model.

    Note:
        This function is used internally within the HuggingFaceTransformersDecoder
        class to facilitate the integration of various model architectures.
    """
    if hasattr(model, "transformer"):
        network = model.transformer
    elif hasattr(model, "gpt_neox"):
        network = model.gpt_neox
    elif hasattr(model, "model"):
        network = model.model
    else:
        raise Exception("Can not find the network attribute")

    return network


def get_hugging_face_model_lm_head(model):
    """
    Get the language model head from a Hugging Face Transformers model.

    This function retrieves the language model head from a given Transformers model.
    The function checks for the presence of the `lm_head` or `embed_out` attribute
    in the model and returns the appropriate head. If neither attribute is found,
    an exception is raised.

    Args:
        model: A Hugging Face Transformers model instance from which to extract
            the language model head.

    Returns:
        The language model head of the specified model.

    Raises:
        Exception: If neither `lm_head` nor `embed_out` attributes can be found
            in the model.

    Examples:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> lm_head = get_hugging_face_model_lm_head(model)
        >>> print(lm_head)  # This will print the language model head of the model.

    Note:
        Ensure that the model is a valid Hugging Face Transformers model instance
        before calling this function.
    """
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    elif hasattr(model, "embed_out"):
        lm_head = model.embed_out
    else:
        raise Exception("Can not find the LM head attribute")

    return lm_head
