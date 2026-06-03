"""Tests for espnet2/speechlm/dataloader/multimodal_loader/dialogue_loader.py."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from espnet2.speechlm.dataloader.multimodal_loader.dialogue_loader import (
    DialogueReader,
    validate_and_process_messages,
)

# ---------- validate_and_process_messages ----------


class TestValidateAndProcessMessages:
    def test_validate_text_valid(self):
        messages = [["user", "text", "hello world"]]
        result = validate_and_process_messages(messages, "test_key")
        assert len(result) == 1
        assert result[0] == ("user", "text", "hello world")

    def test_validate_multiple_messages(self):
        messages = [
            ["user", "text", "hi"],
            ["assistant", "text", "hello"],
            ["user", "text", "bye"],
        ]
        result = validate_and_process_messages(messages, "test_key")
        assert len(result) == 3
        assert result[0] == ("user", "text", "hi")
        assert result[1] == ("assistant", "text", "hello")
        assert result[2] == ("user", "text", "bye")

    def test_validate_empty_messages(self):
        result = validate_and_process_messages([], "test_key")
        assert result == []

    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_validate_all_roles(self, role):
        messages = [[role, "text", "content"]]
        result = validate_and_process_messages(messages, "test_key")
        assert result[0][0] == role

    def test_validate_invalid_role(self):
        messages = [["invalid_role", "text", "content"]]
        with pytest.raises(AssertionError, match="Invalid role"):
            validate_and_process_messages(messages, "test_key")

    def test_validate_invalid_modality(self):
        messages = [["user", "invalid_mod", "content"]]
        with pytest.raises(AssertionError, match="Invalid modality"):
            validate_and_process_messages(messages, "test_key")

    def test_validate_wrong_length(self):
        messages = [["user", "text"]]  # 2 elements instead of 3
        with pytest.raises(AssertionError, match="expected 3 elements"):
            validate_and_process_messages(messages, "test_key")

    def test_validate_non_list(self):
        with pytest.raises(AssertionError, match="expected list"):
            validate_and_process_messages("not a list", "test_key")

    def test_validate_non_string_text(self):
        messages = [["user", "text", 12345]]
        with pytest.raises(AssertionError, match="expected string"):
            validate_and_process_messages(messages, "test_key")

    def test_validate_unsupported_modality(self):
        messages = [["user", "image", "path/to/img.png"]]
        with pytest.raises(ValueError, match="not supported yet"):
            validate_and_process_messages(messages, "test_key")

    def test_validate_audio_mono(self):
        mono_data = np.zeros(16000, dtype=np.float32)
        with patch("soundfile.read", return_value=(mono_data, 16000)):
            messages = [["user", "audio", "/fake/audio.wav"]]
            result = validate_and_process_messages(messages, "test_key")
        audio_array, sr = result[0][2]
        assert sr == 16000
        assert audio_array.shape == (1, 16000)

    def test_validate_audio_stereo(self):
        stereo_data = np.zeros((16000, 2), dtype=np.float32)
        with patch("soundfile.read", return_value=(stereo_data, 16000)):
            messages = [["user", "audio", "/fake/audio.wav"]]
            result = validate_and_process_messages(messages, "test_key")
        audio_array, sr = result[0][2]
        assert sr == 16000
        assert audio_array.shape == (2, 16000)


# ---------- DialogueReader ----------


class TestDialogueReader:
    def _write_dialogue_file(self, tmp_path, dialogues):
        f = tmp_path / "dialogue.jsonl"
        lines = [json.dumps(d) for d in dialogues]
        f.write_text("\n".join(lines) + "\n")
        return str(f)

    def test_reader_basic(self, tmp_path):
        dialogues = [
            {
                "example_id": "dlg1",
                "messages": [["user", "text", "hello"]],
            },
            {
                "example_id": "dlg2",
                "messages": [["assistant", "text", "hi"]],
            },
        ]
        f = self._write_dialogue_file(tmp_path, dialogues)
        reader = DialogueReader(f)
        assert len(reader) == 2
        assert "dlg1" in reader
        assert "dlg2" in reader

    def test_reader_valid_ids(self, tmp_path):
        dialogues = [
            {
                "example_id": "dlg1",
                "messages": [["user", "text", "hello"]],
            },
            {
                "example_id": "dlg2",
                "messages": [["user", "text", "world"]],
            },
        ]
        f = self._write_dialogue_file(tmp_path, dialogues)
        reader = DialogueReader(f, valid_ids=["dlg1"])
        assert len(reader) == 1
        assert "dlg1" in reader
        assert "dlg2" not in reader

    def test_reader_invalid_format(self, tmp_path):
        # Missing "example_id"
        dialogues = [{"messages": [["user", "text", "hello"]]}]
        f = self._write_dialogue_file(tmp_path, dialogues)
        with pytest.raises(ValueError, match="invalid"):
            DialogueReader(f)

    def test_reader_contains(self, tmp_path):
        dialogues = [
            {
                "example_id": "dlg1",
                "messages": [["user", "text", "hello"]],
            },
        ]
        f = self._write_dialogue_file(tmp_path, dialogues)
        reader = DialogueReader(f)
        assert "dlg1" in reader
        assert "nonexistent" not in reader

    def test_reader_keys_values_items(self, tmp_path):
        dialogues = [
            {
                "example_id": "dlg1",
                "messages": [["user", "text", "a"]],
            },
            {
                "example_id": "dlg2",
                "messages": [["assistant", "text", "b"]],
            },
        ]
        f = self._write_dialogue_file(tmp_path, dialogues)
        reader = DialogueReader(f)
        assert set(reader.keys()) == {"dlg1", "dlg2"}
        assert len(list(reader.values())) == 2
        assert len(list(reader.items())) == 2

    def test_reader_getitem_validates(self, tmp_path):
        dialogues = [
            {
                "example_id": "dlg1",
                "messages": [["user", "text", "hello"]],
            },
        ]
        f = self._write_dialogue_file(tmp_path, dialogues)
        reader = DialogueReader(f)
        result = reader["dlg1"]
        assert isinstance(result, list)
        assert result[0] == ("user", "text", "hello")
