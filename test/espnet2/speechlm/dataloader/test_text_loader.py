"""Tests for text_loader.py — TextReader."""

import json

from espnet2.speechlm.dataloader.multimodal_loader.text_loader import TextReader

# ---------- TextReader (plain format) ----------


class TestTextReaderPlain:
    def test_plain_basic(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("utt001 hello world\nutt002 foo bar baz\n")
        reader = TextReader(str(f))
        assert reader["utt001"] == "hello world"
        assert reader["utt002"] == "foo bar baz"

    def test_plain_valid_ids(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("utt001 hello world\nutt002 foo bar\nutt003 baz qux\n")
        reader = TextReader(str(f), valid_ids=["utt001", "utt003"])
        assert "utt001" in reader
        assert "utt002" not in reader
        assert "utt003" in reader
        assert len(reader) == 2

    def test_plain_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("\nutt001 hello\n\nutt002 world\n\n")
        reader = TextReader(str(f))
        assert len(reader) == 2

    def test_plain_single_word_skipped(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("utt001 hello world\njust_id_no_text\nutt002 foo\n")
        reader = TextReader(str(f))
        assert "utt001" in reader
        assert "just_id_no_text" not in reader
        assert "utt002" in reader
        assert len(reader) == 2

    def test_plain_contains(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("utt001 hello world\n")
        reader = TextReader(str(f))
        assert "utt001" in reader
        assert "nonexistent" not in reader

    def test_plain_keys_values_items(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("utt001 hello\nutt002 world\n")
        reader = TextReader(str(f))
        assert set(reader.keys()) == {"utt001", "utt002"}
        assert set(reader.values()) == {"hello", "world"}
        assert set(reader.items()) == {("utt001", "hello"), ("utt002", "world")}

    def test_plain_len(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("a hello\nb world\nc foo\n")
        reader = TextReader(str(f))
        assert len(reader) == 3


# ---------- TextReader (JSONL format) ----------


class TestTextReaderJsonl:
    def test_jsonl_basic(self, tmp_path):
        f = tmp_path / "text.jsonl"
        lines = [
            json.dumps({"id": "utt001", "text": "hello world"}),
            json.dumps({"id": "utt002", "text": "foo bar"}),
        ]
        f.write_text("\n".join(lines) + "\n")
        reader = TextReader(str(f))
        assert reader["utt001"] == "hello world"
        assert reader["utt002"] == "foo bar"

    def test_jsonl_valid_ids(self, tmp_path):
        f = tmp_path / "text.jsonl"
        lines = [
            json.dumps({"id": "utt001", "text": "a"}),
            json.dumps({"id": "utt002", "text": "b"}),
            json.dumps({"id": "utt003", "text": "c"}),
        ]
        f.write_text("\n".join(lines) + "\n")
        reader = TextReader(str(f), valid_ids=["utt001", "utt003"])
        assert len(reader) == 2
        assert "utt002" not in reader

    def test_jsonl_missing_keys_skipped(self, tmp_path):
        f = tmp_path / "text.jsonl"
        lines = [
            json.dumps({"id": "utt001", "text": "hello"}),
            json.dumps({"only_id": "utt002"}),  # missing "text"
            json.dumps({"id": "utt003", "text": "world"}),
        ]
        f.write_text("\n".join(lines) + "\n")
        reader = TextReader(str(f))
        assert len(reader) == 2
        assert "utt002" not in reader

    def test_jsonl_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "text.jsonl"
        content = (
            "\n"
            + json.dumps({"id": "utt001", "text": "a"})
            + "\n\n"
            + json.dumps({"id": "utt002", "text": "b"})
            + "\n\n"
        )
        f.write_text(content)
        reader = TextReader(str(f))
        assert len(reader) == 2
