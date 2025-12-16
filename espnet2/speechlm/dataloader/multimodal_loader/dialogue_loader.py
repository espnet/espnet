#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Dialogue data loading utilities supporting multimodal conversation formats."""

import json
from pathlib import Path
from typing import Any, ItemsView, KeysView, List, Optional, Tuple, Union, ValuesView

import numpy as np
import soundfile as sf


class DialogueReader:

    VALID_ROLES = {"user", "assistant", "system"}
    VALID_MODALITIES = {"text", "audio", "image", "video", "toolcall"}

    def __init__(self, dialogue_file: str, valid_ids: Optional[List[str]] = None):
        self.dialogues = {}

        valid_ids = set(valid_ids) if valid_ids is not None else None
        for idx, line in enumerate(open(dialogue_file)):
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

        assert isinstance(messages, list), f"Invalid messages for {key}: expected list"

        validated = []
        for i, msg in enumerate(messages):
            role, modality, content = msg

            assert role in self.VALID_ROLES, (
                f"Invalid role '{role}' at index {i} for {key}: "
                f"must be one of {self.VALID_ROLES}"
            )

            assert modality in self.VALID_MODALITIES, (
                f"Invalid modality '{modality}' at index {i} for {key}: "
                f"must be one of {self.VALID_MODALITIES}"
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

            validated.append((role, modality, processed_content))

        return validated

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
