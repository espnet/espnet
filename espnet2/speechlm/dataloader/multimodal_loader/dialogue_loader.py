#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Dialogue data loading utilities supporting multimodal conversation formats."""

import json
from pathlib import Path
from typing import (
    Any,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Optional,
    Tuple,
    Union,
    ValuesView,
)

import numpy as np
import soundfile as sf

from espnet2.speechlm.dataloader.multimodal_loader.text_loader import ArkiveTextReader

VALID_ROLES = {"user", "assistant", "system"}
VALID_MODALITIES = {"text", "audio", "image", "video", "toolcall"}


def validate_and_process_messages(
    messages: List[Any], key: str
) -> List[Tuple[str, str, Union[str, Tuple[np.ndarray, int]]]]:
    """Validate and process dialogue messages.

    Args:
        messages: List of message tuples (role, modality, content).
        key: The example ID for error messages.

    Returns:
        List of validated tuples where each tuple is (role, modality, content).
        Content format depends on modality:
        - text: string
        - audio: (audio_array, sample_rate) where audio_array has shape
                [num_channels, num_samples]
    """
    assert isinstance(messages, list), f"Invalid messages for {key}: expected list"

    validated = []
    for i, msg in enumerate(messages):
        assert len(msg) == 3, (
            f"Invalid message format at index {i} for {key}: "
            f"expected 3 elements (role, modality, content), got {len(msg)}"
        )
        role, modality, content = msg

        assert role in VALID_ROLES, (
            f"Invalid role '{role}' at index {i} for {key}: "
            f"must be one of {VALID_ROLES}"
        )

        assert modality in VALID_MODALITIES, (
            f"Invalid modality '{modality}' at index {i} for {key}: "
            f"must be one of {VALID_MODALITIES}"
        )

        # Validate and process content based on modality
        if modality == "text":
            assert isinstance(content, str), (
                f"Invalid text content at index {i} for {key}: "
                f"expected string, got {type(content)}"
            )
            processed_content = content
        elif modality == "audio":
            # Load audio file
            audio_path = Path(content)

            # Load audio using soundfile
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")

            # Ensure shape is [num_channels, num_samples]
            if audio_data.ndim == 1:
                # Single channel - add channel dimension
                audio_data = audio_data[np.newaxis, :]
            elif audio_data.ndim == 2:
                # Multi-channel: [samples, channels] -> [channels, samples]
                audio_data = audio_data.T
            else:
                raise ValueError(
                    f"Unexpected audio shape at index {i} for {key}: "
                    f"{audio_data.shape}"
                )

            processed_content = (audio_data, sample_rate)
        else:
            raise ValueError(f"For now {modality} is not supported yet")

        validated.append((role, modality, processed_content))

    return validated


class DialogueReader:

    def __init__(self, dialogue_file: str, valid_ids: Optional[List[str]] = None):
        self.dialogues = {}

        valid_ids = set(valid_ids) if valid_ids is not None else None
        with open(dialogue_file) as f:
            for idx, line in enumerate(f):
                line = json.loads(line)

                if not ("example_id" in line and "messages" in line):
                    raise ValueError(f"Line {idx} of file {dialogue_file} is invalid")

                if valid_ids is not None and line["example_id"] not in valid_ids:
                    continue

                self.dialogues[line["example_id"]] = line["messages"]

    def __getitem__(
        self, key: str
    ) -> List[Tuple[str, str, Union[str, Tuple[np.ndarray, int]]]]:
        """Get dialogue messages by ID with validation and content loading.

        Returns:
            List of tuples where each tuple is (role, modality, content).
            Content format depends on modality:
            - text: string
            - audio: (audio_array, sample_rate) where audio_array has shape
                    [num_channels, num_samples]
        """
        messages = self.dialogues[key]
        return validate_and_process_messages(messages, key)

    def __contains__(self, key: str) -> bool:
        """Check if ID exists."""
        return key in self.dialogues

    def __len__(self) -> int:
        """Return number of dialogues."""
        return len(self.dialogues)

    def keys(self) -> KeysView[str]:
        """Return iterator over IDs."""
        return self.dialogues.keys()

    def values(self) -> ValuesView[List[Any]]:
        """Return iterator over dialogues."""
        # Note: returns raw values without validation
        return self.dialogues.values()

    def items(self) -> ItemsView[str, List[Any]]:
        """Return iterator over (id, dialogue) pairs."""
        # Note: returns raw items without validation
        return self.dialogues.items()


class ArkiveDialogueLoader(ArkiveTextReader):
    """Dict-like lazy dialogue reader using arkive parquets.

    Extends ArkiveTextReader to parse JSON string output into validated
    dialogue messages with the same sanity checks as DialogueReader.

    Args:
        parquet_path: Path to the parquet file containing dialogue metadata.
        valid_ids: List of valid IDs to keep (optional, keeps all if None).
        worker_id: Partition IDs by worker (optional, keeps all if None).
        world_size: Used for worker partitioning.
    """

    def __getitem__(
        self, key: str
    ) -> List[Tuple[str, str, Union[str, Tuple[np.ndarray, int]]]]:
        """Get dialogue messages by ID with validation and content loading.

        Returns:
            List of tuples where each tuple is (role, modality, content).
            Content format depends on modality:
            - text: string
            - audio: (audio_array, sample_rate) where audio_array has shape
                    [num_channels, num_samples]
        """
        # Get compressed text from parent class and parse as JSON
        text = super().__getitem__(key)
        messages = json.loads(text)
        return validate_and_process_messages(messages, key)

    def values(
        self,
    ) -> Iterator[List[Tuple[str, str, Union[str, Tuple[np.ndarray, int]]]]]:
        """Return iterator over validated dialogue values."""
        for key in self.data:
            yield self[key]

    def items(
        self,
    ) -> Iterator[
        Tuple[str, List[Tuple[str, str, Union[str, Tuple[np.ndarray, int]]]]]
    ]:
        """Return iterator over (id, validated_dialogue) pairs."""
        for key in self.data:
            yield key, self[key]
