"""Unit tests for espnet3.components.metrics.versa."""

import json

import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.components.metrics.versa import VersaMetric


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        f.write("\n")  # blank lines are skipped


class TestResolveScoreConfigPath:
    def test_existing_file_is_used_as_is(self, tmp_path):
        config_file = tmp_path / "score.yaml"
        config_file.write_text("- name: signal_metric\n", encoding="utf-8")
        metric = VersaMetric(score_config=str(config_file))

        assert metric._resolve_score_config_path(tmp_path) == config_file

    def test_missing_file_raises(self, tmp_path):
        metric = VersaMetric(score_config=str(tmp_path / "nope.yaml"))

        with pytest.raises(FileNotFoundError):
            metric._resolve_score_config_path(tmp_path)

    def test_inline_list_is_dumped(self, tmp_path):
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        out = metric._resolve_score_config_path(tmp_path)

        assert out == tmp_path / "versa_config.yaml"
        assert yaml.safe_load(out.read_text()) == [{"name": "signal_metric"}]

    def test_omegaconf_container_is_converted(self, tmp_path):
        metric = VersaMetric(
            score_config=OmegaConf.create([{"name": "pseudo_mos", "fs": 16000}])
        )

        out = metric._resolve_score_config_path(tmp_path)

        assert yaml.safe_load(out.read_text()) == [{"name": "pseudo_mos", "fs": 16000}]


class TestAggregate:
    def test_averages_numeric_fields_only(self, tmp_path):
        result_file = tmp_path / "result.json"
        _write_jsonl(
            result_file,
            [
                {"key": "utt1", "mcd": 1.0, "pesq": 2.0, "ok": True},
                {"key": "utt2", "mcd": 2.0, "pesq": 4.0, "ok": False},
            ],
        )

        assert VersaMetric._aggregate(result_file) == {"mcd": 1.5, "pesq": 3.0}

    def test_missing_fields_average_over_present_rows(self, tmp_path):
        result_file = tmp_path / "result.json"
        _write_jsonl(result_file, [{"a": 1.0}, {"a": 2.0, "b": 10.0}])

        assert VersaMetric._aggregate(result_file) == {"a": 1.5, "b": 10.0}


class TestCall:
    def _make_data(self, tmp_path):
        wav = tmp_path / "wav.scp"
        ref = tmp_path / "ref.scp"
        wav.write_text("utt1 a.wav\n", encoding="utf-8")
        ref.write_text("utt1 b.wav\n", encoding="utf-8")
        return {"wav": wav, "ref": ref}

    @pytest.mark.parametrize("missing", ["wav", "ref"])
    def test_missing_required_input_raises(self, tmp_path, missing):
        data = self._make_data(tmp_path)
        data.pop(missing)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        with pytest.raises(KeyError, match=missing):
            metric(data, "test", tmp_path / "inference")

    def test_missing_optional_text_input_raises(self, tmp_path):
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], text_key="text")

        with pytest.raises(KeyError, match="text"):
            metric(self._make_data(tmp_path), "test", tmp_path / "inference")

    def test_builds_command_and_returns_averages(self, tmp_path, monkeypatch):
        recorded = {}

        def fake_run(cmd, check):
            recorded["cmd"] = cmd
            recorded["check"] = check
            output_file = cmd[cmd.index("--output_file") + 1]
            _write_jsonl(tmp_path / "out.json", [{"mcd": 3.0}])
            (tmp_path / "out.json").rename(output_file)

        monkeypatch.setattr("espnet3.components.metrics.versa.subprocess.run", fake_run)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], use_gpu=False)
        output_dir = tmp_path / "inference"

        # Keyword call pins the BaseMetric contract (data, test_name, output_dir).
        averages = metric(self._make_data(tmp_path), "test", output_dir=output_dir)

        assert averages == {"mcd": 3.0}
        assert recorded["check"] is True
        assert recorded["cmd"][:3] == ["python", "-m", "versa.bin.scorer"]
        assert "--use_gpu" not in recorded["cmd"]
        assert "--text" not in recorded["cmd"]

        eval_dir = output_dir / "test" / "scoring" / "versa_eval"
        assert json.loads((eval_dir / "avg_result.json").read_text()) == {"mcd": 3.0}

    def test_use_gpu_and_text_are_forwarded(self, tmp_path, monkeypatch):
        recorded = {}

        def fake_run(cmd, check):
            recorded["cmd"] = cmd
            output_file = cmd[cmd.index("--output_file") + 1]
            _write_jsonl(tmp_path / "out.json", [{"mcd": 1.0}])
            (tmp_path / "out.json").rename(output_file)

        monkeypatch.setattr("espnet3.components.metrics.versa.subprocess.run", fake_run)
        data = self._make_data(tmp_path)
        text = tmp_path / "text"
        text.write_text("utt1 hello\n", encoding="utf-8")
        data["text"] = text
        metric = VersaMetric(
            score_config=[{"name": "signal_metric"}], text_key="text", use_gpu=True
        )

        metric(data, "test", tmp_path / "inference")

        assert "--use_gpu" in recorded["cmd"]
        assert recorded["cmd"][recorded["cmd"].index("--text") + 1] == str(text)


class TestSummarize:
    def test_logs_plain_metrics(self, caplog):
        with caplog.at_level("INFO", logger="espnet3.components.metrics.versa"):
            VersaMetric.summarize({"mcd": 1.2345}, "test")

        assert "VERSA scores - test" in caplog.text
        assert "mcd" in caplog.text

    def test_logs_wer_and_cer_component_groups(self, caplog):
        scores = {
            "mcd": 1.0,
            "espnet_wer_delete": 1.0,
            "espnet_wer_insert": 1.0,
            "espnet_wer_replace": 2.0,
            "espnet_wer_equal": 96.0,
            "espnet_cer_delete": 0.0,
            "espnet_cer_insert": 0.0,
            "espnet_cer_replace": 1.0,
            "espnet_cer_equal": 99.0,
        }

        with caplog.at_level("INFO", logger="espnet3.components.metrics.versa"):
            VersaMetric.summarize(scores, "test")

        assert "WER components" in caplog.text
        assert "CER components" in caplog.text
        # (1 + 1 + 2) / 100 -> 4.00%
        assert "4.00%" in caplog.text
        assert "1.00%" in caplog.text
