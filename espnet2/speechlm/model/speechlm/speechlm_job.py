#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SpeechLM job template implementation for multimodal language modeling."""

import re
from typing import Any, Callable, Dict

import numpy as np
import torch

from espnet2.speechlm.model.abs_job import AbsJobTemplate

# Main speechlm model
from espnet2.speechlm.model.speechlm.lm.parallel import ParallelHFModel

# Multimodal IOs
from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO
from espnet2.speechlm.model.speechlm.multimodal_io.audio import (
    ContinuousAudioIO,
    DiscreteAudioIO,
)
from espnet2.speechlm.model.speechlm.multimodal_io.text import HuggingFaceTextIO
from espnet2.speechlm.model.speechlm.task_conf_speechlm import SPEECHLM_TASK_CONFIGS
from espnet2.speechlm.utils.data import pad_list

_multimodal_ios = {
    "text": HuggingFaceTextIO,
    "discrete_audio": DiscreteAudioIO,
    "continuous_audio": ContinuousAudioIO,
}

_lms = {"parallel": ParallelHFModel}


class SpeechLMJobTemplate(AbsJobTemplate):
    """Job template for SpeechLM training tasks.

    This class implements the specific model and data processing
    configurations for speech language modeling tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the SpeechLM job template.

        Args:
            config: Dictionary containing job configuration parameters.
        """
        super().__init__(config)

        # (1) keep other configs
        self.config = config

        # (2) build tokenizers and vocabulary
        io_config = config["multimodal_io"]
        self.multimodal_io = dict()
        for io_name, io_kwargs in io_config.items():
            multimodal_io_class = _multimodal_ios[io_name]
            assert issubclass(multimodal_io_class, AbsIO)
            self.multimodal_io[io_name] = multimodal_io_class(**io_kwargs)

        self.vocab, self.vocab_intervals = self._build_vocabulary()

    def _build_vocabulary(self, num_special_tokens=256):
        """Build unified vocabulary from special tokens and multimodal IOs.

        Reserves fixed slots for special tokens then adds tokens from discrete IOs.
        Returns vocabulary list and interval mappings for each modality.
        """
        # (1) Initial special token. We keep a fixed number of slots
        vocab_intervals = {"special_token": [(0, num_special_tokens)]}
        vocab = [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|eot|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|text|>",
            "<|audio|>",
            "<|image|>",
            "<|video|>",
            "<|toolcall|>",
        ]
        while len(vocab) < num_special_tokens:
            vocab.append(f"<|unused_{len(vocab)}|>")

        # (2) add vocabulary from each discrete multimodal IO.
        start = num_special_tokens
        for io_name, io in self.multimodal_io.items():
            if io.is_discrete:
                vocab.extend(io.get_vocabulary())
                vocab_intervals[io_name] = [
                    (start + this_start, start + this_end)
                    for this_start, this_end in io.get_stream_interval()
                ]
                start = len(vocab)

        assert len(vocab) == len(set(vocab)), "There are duplicated tokens in the vocab"

        return vocab, vocab_intervals

    def build_preprocessor(self) -> Callable:
        """Build the data collation function for SpeechLM.

        Returns:
            A callable function for collating SpeechLM batch data.
        """

        processor_config = self.config["preprocessor"]
        multimodal_io = {
            io_name: io.copy_for_worker() for io_name, io in self.multimodal_io.items()
        }
        return SpeechLMPreprocessor(
            multimodal_io=multimodal_io,
            vocab=self.vocab,
            vocab_intervals=self.vocab_intervals,
            audio_input=processor_config["audio_input"],
            audio_output=processor_config["audio_output"],
            loss_region=processor_config["loss_region"],
            batchfy_method=self.config["data_loading"].get("batchfy_method", "bucket"),
        )

    def build_model(self) -> torch.nn.Module:
        """Build the SpeechLM model.

        Returns:
            A SpeechLM model instance.
        """

        model_config = self.config["model"]
        model_class = _lms[model_config["model_choice"]]

        model = model_class(
            model_hf_tag=model_config["model_hf_tag"],
            multimodal_io=self.multimodal_io,
            vocab_intervals=self.vocab_intervals,
            **model_config["model_conf"],
        )

        if model_config.get("activation_checkpointing", False):
            model.gradient_checkpointing_enable()

        return model


class SpeechLMPreprocessor:
    """Preprocessor for SpeechLM data handling.

    Converts raw data into model-ready format with tokenization,
    padding, and loss mask generation for multimodal sequences.
    """

    def __init__(
        self,
        multimodal_io,
        vocab,
        vocab_intervals,
        audio_input: str = "continuous_audio",
        audio_output: str = "discrete_audio",
        loss_region: str = "assistant",
        batchfy_method: str = "bucket",
    ):

        # (1) keep all multimodal_io
        self.multimodal_io = multimodal_io
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.loss_region = loss_region
        self.batchfy_method = batchfy_method

        # (2) vocabulary
        self.vocab = vocab
        self.vocab_intervals = vocab_intervals
        self.pad_id = self.vocab.index("<|pad|>")

        possible_num_stream = [
            io.num_stream() for io in multimodal_io.values() if io.is_discrete
        ]
        if len(possible_num_stream) == 0:
            raise ValueError("You should have at least one discrete multimodal IO")
        self.num_stream = max(possible_num_stream)

    def find_length(self, key, data_dict):
        """Quickly compute sequence length without full preprocessing.

        Counts tokens for BOS, role/modality markers, content, and EOS/EOT.
        Used for efficient batch construction.
        """
        task, _, _ = key
        messages = self._apply_chat_template(task, data_dict)

        # (1) <bos>
        length = 1

        # (2) each message, consider role, modality and end of <eot>/<eos>
        for _, this_io, this_data in messages:
            length += 3
            length += self.multimodal_io[this_io].find_length(this_data)

        return length

    def collate_fn(self, data_lst):
        """Batch multiple samples for training.

        Processes each sample, pads sequences to same length, and organizes
        continuous features by modality. Returns dict ready for model forward.
        """
        if self.batchfy_method != "bucket":
            raise NotImplementedError("Only bucket collate function is implemented")

        data_dicts = [self.preprocessing(key, data_dict) for key, data_dict in data_lst]

        seqs, conti_feats, loss_masks = [], [], []
        for bidx, data_dict in enumerate(data_dicts):
            seqs.append(data_dict["sequence"])
            loss_masks.append(data_dict["loss_mask"])

            for conti_feat in data_dict["conti_feats"]:
                conti_feats.append((bidx,) + conti_feat)

        seqs, _ = pad_list(seqs)
        loss_masks, _ = pad_list(loss_masks)

        conti_feats_dict = dict()
        for bidx, this_io, start, length, feat in conti_feats:
            if this_io not in conti_feats_dict:
                conti_feats_dict[this_io] = [[], []]
            conti_feats_dict[this_io][0].append((bidx, start, length))
            conti_feats_dict[this_io][1].append(feat)

        for io_dict in conti_feats_dict.values():
            io_dict[1] = pad_list(io_dict[1])

        keys = [key for key, _ in data_lst]

        return {
            "key": keys,
            "seqs": seqs,
            "conti_feats": conti_feats_dict,
            "loss_masks": loss_masks,
        }

    def preprocessing(self, key, data_dict):
        """Convert single raw data dict into training-ready format.

        Applies chat template, tokenizes content, adds special tokens,
        and creates loss masks. Returns dict with sequences and features.
        """
        # (1) convert to messages
        task, _, _ = key
        messages = self._apply_chat_template(task, data_dict)

        # (2) initialize
        seq = [self.special_token("<|bos|>")]
        conti_feats = list()
        loss_masks = [self.special_mask(0.0)]
        accum_length = 1

        # (3) loop on each message
        # Determine where to place EOT tokens (when consecutive msgs have same role)
        apply_eots = [
            msg1[0] == msg2[0] for msg1, msg2 in zip(messages[:-1], messages[1:])
        ] + [False]
        for apply_eot, (role, this_io, this_data) in zip(apply_eots, messages):
            apply_loss = float(role == "assistant" or self.loss_region == "all")
            special_mask = self.special_mask(apply_loss)

            # (3.1) role and modality
            seq.append(self.special_token(f"<|{role}|>"))
            loss_masks.append(special_mask)

            modality = self.multimodal_io[this_io].modality
            seq.append(self.special_token(f"<|{modality}|>"))
            loss_masks.append(special_mask)

            accum_length += 2

            # (3.2) the exact data processing
            this_seq, conti_feat, loss_mask = self.multimodal_io[this_io].preprocess(
                this_data
            )
            assert this_seq.shape == loss_mask.shape

            # (3.3) this_seq - adjust token IDs and pad to match stream count
            if self.multimodal_io[this_io].is_discrete:
                modality_bias = self.vocab_intervals[this_io][0][0]
                this_seq = np.where(
                    this_seq == self.pad_id, self.pad_id, this_seq + modality_bias
                )
            # Pad to num_stream if current IO has fewer streams
            if this_seq.shape[1] < self.num_stream:
                pad_size = self.num_stream - this_seq.shape[1]
                this_seq = np.pad(this_seq, ((0, 0), (0, pad_size)))
            seq.append(this_seq)

            # (3.4) conti_feats
            if conti_feat is not None:
                length, feat = conti_feat
                conti_feats.append((this_io, accum_length, length, feat))

            # (3.5) loss_mask - pad and apply based on role
            # Pad loss mask to match num_stream dimensions
            if loss_mask.shape[1] < self.num_stream:
                pad_size = self.num_stream - loss_mask.shape[1]
                loss_mask = np.pad(loss_mask, ((0, 0), (0, pad_size)))
            loss_masks.append(loss_mask * apply_loss)

            accum_length += this_seq.shape[0]

            # (3.6) <eot> or <eos>
            if apply_eot:
                seq.append(self.special_token("<|eot|>"))
            else:
                seq.append(self.special_token("<|eos|>"))
            loss_masks.append(special_mask)
            accum_length += 1

        # (4) concat
        seq = np.concatenate(seq, axis=0)
        loss_mask = np.concatenate(loss_masks, axis=0)

        # TODO(speechlm): Add CFG here
        data = {
            "sequence": seq,
            "conti_feats": conti_feats,
            "loss_mask": loss_mask,
        }

        return data

    def diagnose(self, data):
        """Print human-readable representation of processed data for debugging.

        Shows tokens, loss masks, and continuous feature info frame by frame.
        """
        seq = data["sequence"]
        loss_mask = data["loss_mask"]
        conti_feats = data["conti_feats"]

        for i, (s, m) in enumerate(zip(seq, loss_mask)):
            s = [self.vocab[s] for s in s.tolist()]
            m = m.tolist()
            print(f"Frame {i} | token: {s} | weight: {m}")

        for this_io, conti_start, length, feat in conti_feats:
            print(
                f"Conti feats: modality={this_io}, conti_feat={conti_start}, "
                f"length={length}, feat={feat.shape}"
            )

    def special_mask(self, value):
        """Create loss mask for special tokens (1 frame, multi-stream).

        Only first stream has the actual value, others are zero.
        """
        retval = np.zeros((1, self.num_stream)).astype(np.float32)
        retval[0, 0] = value
        return retval

    def special_token(self, token):
        """Convert special token string to multi-stream token array.

        Places token ID in first stream, padding tokens in other streams.
        """
        num_special_token = self.vocab_intervals["special_token"][0][1]
        special_tokens = self.vocab[:num_special_token]
        token_id = special_tokens.index(token)
        retval = np.ones((1, self.num_stream)).astype(np.int64) * self.pad_id
        retval[0, 0] = token_id
        return retval

    def _apply_chat_template(self, task, data_dict):
        """Convert data dict to list of (role, io_type, data) messages.

        Either uses provided dialogue or constructs from task template.
        Determines appropriate IO type based on role and data entry name.
        """
        if "dialogue" in data_dict:
            if len(data_dict) != 1:
                raise ValueError(
                    "If dialogue exist, there should be no more other entries"
                )
            return data_dict["dialogue"]
        else:
            task_config = SPEECHLM_TASK_CONFIGS[task]
            messages = list()
            for role, entry in task_config:
                # Select IO type based on entry name and role
                if bool(re.match(r"^audio", entry)):
                    # User/system use input audio IO, assistant uses output audio IO
                    if role == "user" or role == "system":
                        this_io = self.audio_input
                    else:
                        this_io = self.audio_output
                elif bool(re.match(r"^text", entry)):
                    this_io = "text"
                else:
                    raise ValueError(f"Not supported data entry in template: {entry}")

                this_data = data_dict[entry]
                message = (role, this_io, this_data)
                messages.append(message)
            return messages
