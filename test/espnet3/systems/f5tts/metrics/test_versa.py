"""Unit tests for espnet3.systems.f5tts.metrics.versa."""

import io
import json
import logging
import pathlib
import re
import subprocess
import sys

import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.systems.f5tts.metrics.versa import VersaMetric


def _write_jsonl(path, records):
    """Write ``records`` as JSON lines, ending with a blank line."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        f.write("\n")  # blank lines are skipped


def _ops(prefix, *, delete, insert, replace, equal):
    """One VERSA record of edit-operation counts under ``prefix``."""
    return {
        f"{prefix}_delete": delete,
        f"{prefix}_insert": insert,
        f"{prefix}_replace": replace,
        f"{prefix}_equal": equal,
    }


class FakePopen:
    """Stands in for the scorer process: prints `lines`, writes `records`."""

    def __init__(self, lines=(), records=({"mcd": 3.0},), returncode=0):
        """Store the output lines, result records and exit code to replay."""
        self._lines = list(lines)
        self._records = list(records)
        self._returncode = returncode
        self.calls = []

    def __call__(self, cmd, **kwargs):
        """Record the command, write the result file and expose the output."""
        self.calls.append(cmd)
        output_file = cmd[cmd.index("--output_file") + 1]
        _write_jsonl(pathlib.Path(output_file), self._records)
        self.stdout = io.StringIO("".join(line + "\n" for line in self._lines))
        return self

    def wait(self):
        """Return the configured exit code."""
        return self._returncode


class TestResolveScoreConfigPath:
    """``_resolve_score_config_path`` with a file path or an inline list."""

    def test_existing_file_is_used_as_is(self, tmp_path):
        """An existing config file is passed to VERSA unchanged."""
        config_file = tmp_path / "score.yaml"
        config_file.write_text("- name: signal_metric\n", encoding="utf-8")
        metric = VersaMetric(score_config=str(config_file))

        assert metric._resolve_score_config_path(tmp_path) == config_file

    def test_missing_file_raises(self, tmp_path):
        """A config path that does not exist raises."""
        metric = VersaMetric(score_config=str(tmp_path / "nope.yaml"))

        with pytest.raises(FileNotFoundError):
            metric._resolve_score_config_path(tmp_path)

    def test_inline_list_is_dumped(self, tmp_path):
        """An inline metric list is written to ``versa_config.yaml``."""
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        out = metric._resolve_score_config_path(tmp_path)

        assert out == tmp_path / "versa_config.yaml"
        assert yaml.safe_load(out.read_text()) == [{"name": "signal_metric"}]

    def test_omegaconf_container_is_converted(self, tmp_path):
        """An OmegaConf list is converted before it is written."""
        metric = VersaMetric(
            score_config=OmegaConf.create([{"name": "pseudo_mos", "fs": 16000}])
        )

        out = metric._resolve_score_config_path(tmp_path)

        assert yaml.safe_load(out.read_text()) == [{"name": "pseudo_mos", "fs": 16000}]


class TestAggregate:
    """``_aggregate`` averages the per-utterance records."""

    def test_averages_numeric_fields_only(self, tmp_path):
        """Numbers are averaged; booleans and strings are left out."""
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
        """A field averages over the records that have it."""
        result_file = tmp_path / "result.json"
        _write_jsonl(result_file, [{"a": 1.0}, {"a": 2.0, "b": 10.0}])

        assert VersaMetric._aggregate(result_file) == {"a": 1.5, "b": 10.0}


class TestPooledErrorRates:
    """WER/CER pool edit-op counts over the corpus; the mean of rates is wrong."""

    def test_pools_error_counts_instead_of_averaging(self, tmp_path):
        # utt1: D=1 S=0 I=0 C=1 (50%); utt2: D=0 S=1 I=0 C=97 (1.02%).
        # Pooled: errors 2 over reference length 100 -> 2.00%, not 25.51%.
        """WER is pooled edit counts over the reference length, not a mean of rates."""
        result_file = tmp_path / "result.json"
        _write_jsonl(
            result_file,
            [
                _ops("fwhisper_wer", delete=1, insert=0, replace=0, equal=1),
                _ops("fwhisper_wer", delete=0, insert=0, replace=1, equal=97),
            ],
        )

        scores = VersaMetric._aggregate(result_file)

        assert scores["fwhisper_wer"] == pytest.approx(2.0)
        # The per-op component means are still reported.
        assert scores["fwhisper_wer_equal"] == pytest.approx(49.0)
        assert scores["fwhisper_wer_delete"] == pytest.approx(0.5)

    def test_pools_cer_too(self, tmp_path):
        # Pooled D=2 I=3 S=0 C=95: errors 5 over reference length 97.
        """CER is pooled the same way as WER."""
        result_file = tmp_path / "result.json"
        _write_jsonl(
            result_file,
            [
                _ops("fwhisper_cer", delete=2, insert=0, replace=0, equal=8),
                _ops("fwhisper_cer", delete=0, insert=3, replace=0, equal=87),
            ],
        )

        assert VersaMetric._aggregate(result_file)["fwhisper_cer"] == pytest.approx(
            5.1546
        )

    def test_excludes_insertions_from_the_denominator(self, tmp_path):
        """Insertions are errors but not reference tokens.

        Pooled D=3 I=50 S=7 C=40: 60 errors over 50 reference words is 120%,
        which is correct; the alignment-length denominator would say 60%.
        VERSA's own fwhisper_wer asserts delete + replace + equal == len(ref).
        """
        result_file = tmp_path / "result.json"
        _write_jsonl(
            result_file,
            [
                _ops("fwhisper_wer", delete=2, insert=30, replace=3, equal=15),
                _ops("fwhisper_wer", delete=1, insert=20, replace=4, equal=25),
            ],
        )

        assert VersaMetric._aggregate(result_file)["fwhisper_wer"] == pytest.approx(
            120.0
        )

    def test_skips_the_rate_when_the_reference_is_empty(self, tmp_path):
        """No reference tokens: the rate is undefined, not a ZeroDivisionError."""
        result_file = tmp_path / "result.json"
        _write_jsonl(
            result_file, [_ops("fwhisper_wer", delete=0, insert=5, replace=0, equal=0)]
        )

        scores = VersaMetric._aggregate(result_file)

        assert "fwhisper_wer" not in scores
        assert scores["fwhisper_wer_insert"] == pytest.approx(5.0)

    def test_leaves_other_metrics_alone(self, tmp_path):
        """Metrics without edit counts keep their plain mean."""
        result_file = tmp_path / "result.json"
        _write_jsonl(result_file, [{"utmos": 4.0}, {"utmos": 4.5}])

        scores = VersaMetric._aggregate(result_file)

        assert scores == {"utmos": 4.25}

    def test_find_prefix_requires_all_four_ops(self):
        """An edit-count group is found only when all four counts are present."""
        partial = {"fwhisper_wer_delete": 1, "fwhisper_wer_equal": 9}
        assert VersaMetric._find_prefix(partial, "wer") is None
        complete = _ops("fwhisper_wer", delete=1, insert=0, replace=0, equal=9)
        assert VersaMetric._find_prefix(complete, "wer") == "fwhisper_wer_"


class TestCall:
    """``VersaMetric.__call__`` builds and runs the scorer command."""

    def _make_data(self, tmp_path):
        """Write one-line ``wav`` and ``ref`` SCP files and return their paths."""
        wav = tmp_path / "wav.scp"
        ref = tmp_path / "ref.scp"
        wav.write_text("utt1 a.wav\n", encoding="utf-8")
        ref.write_text("utt1 b.wav\n", encoding="utf-8")
        return {"wav": wav, "ref": ref}

    @pytest.mark.parametrize("missing", ["wav", "ref"])
    def test_missing_required_input_raises(self, tmp_path, missing):
        """A missing ``wav`` or ``ref`` input raises a ``KeyError``."""
        data = self._make_data(tmp_path)
        data.pop(missing)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        with pytest.raises(KeyError, match=missing):
            metric(data, "test", tmp_path / "inference")

    def test_missing_optional_text_input_raises(self, tmp_path):
        """A configured ``text_key`` without its input raises."""
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], text_key="text")

        with pytest.raises(KeyError, match="text"):
            metric(self._make_data(tmp_path), "test", tmp_path / "inference")

    def test_builds_command_and_returns_averages(self, tmp_path, monkeypatch):
        """The scorer runs under this interpreter and its averages are returned."""
        scorer = FakePopen(lines=["scoring utt1"], records=[{"mcd": 3.0}])
        monkeypatch.setattr(
            "espnet3.systems.f5tts.metrics.versa.subprocess.Popen", scorer
        )
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], use_gpu=False)
        output_dir = tmp_path / "inference"

        # Keyword call pins the BaseMetric contract (data, test_name, output_dir).
        averages = metric(self._make_data(tmp_path), "test", output_dir=output_dir)

        assert averages == {"mcd": 3.0}
        (cmd,) = scorer.calls
        # sys.executable, not "python": under srun or a venv the interpreter
        # running the stage is the only one known to have versa installed.
        assert cmd[:3] == [sys.executable, "-m", "versa.bin.scorer"]
        assert "--use_gpu" not in cmd
        assert "--text" not in cmd

        eval_dir = output_dir / "test" / "scoring" / "versa_eval"
        assert json.loads((eval_dir / "avg_result.json").read_text()) == {"mcd": 3.0}

    def test_use_gpu_and_text_are_forwarded(self, tmp_path, monkeypatch):
        """``use_gpu`` and the transcript file reach the scorer command."""
        scorer = FakePopen(records=[{"mcd": 1.0}])
        monkeypatch.setattr(
            "espnet3.systems.f5tts.metrics.versa.subprocess.Popen", scorer
        )
        data = self._make_data(tmp_path)
        text = tmp_path / "text"
        text.write_text("utt1 hello\n", encoding="utf-8")
        data["text"] = text
        metric = VersaMetric(
            score_config=[{"name": "signal_metric"}], text_key="text", use_gpu=True
        )

        metric(data, "test", tmp_path / "inference")

        (cmd,) = scorer.calls
        assert "--use_gpu" in cmd
        assert cmd[cmd.index("--text") + 1] == str(text)

    def test_non_zero_exit_raises(self, tmp_path, monkeypatch):
        """A scorer that exits non-zero raises ``CalledProcessError``."""
        scorer = FakePopen(returncode=1)
        monkeypatch.setattr(
            "espnet3.systems.f5tts.metrics.versa.subprocess.Popen", scorer
        )
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        with pytest.raises(subprocess.CalledProcessError):
            metric(self._make_data(tmp_path), "test", tmp_path / "inference")

    def test_failed_metric_is_an_error_even_when_the_scorer_exits_zero(
        self, tmp_path, monkeypatch
    ):
        """VERSA drops a metric it cannot load and still exits 0."""
        scorer = FakePopen(
            lines=["Failed to load metric pseudo_mos: No module named 'utmos'"],
            records=[{"mcd": 3.0}],
        )
        monkeypatch.setattr(
            "espnet3.systems.f5tts.metrics.versa.subprocess.Popen", scorer
        )
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        with pytest.raises(RuntimeError, match="Failed to load metric pseudo_mos"):
            metric(self._make_data(tmp_path), "test", tmp_path / "inference")

    def test_metric_that_never_scores_is_an_error(self, tmp_path, monkeypatch):
        """A metric that throws per utterance leaves its key null everywhere."""
        scorer = FakePopen(
            records=[{"mcd": 3.0, "utmos": None}, {"mcd": 1.0, "utmos": None}]
        )
        monkeypatch.setattr(
            "espnet3.systems.f5tts.metrics.versa.subprocess.Popen", scorer
        )
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        with pytest.raises(RuntimeError, match="no value for any utterance: utmos"):
            metric(self._make_data(tmp_path), "test", tmp_path / "inference")

    def test_text_fields_and_sparse_nulls_are_not_failures(self, tmp_path, monkeypatch):
        """Only a key that is null in every record and numeric in none counts."""
        scorer = FakePopen(
            records=[
                {"mcd": 3.0, "utmos": 4.0, "hyp_text": "a"},
                {"mcd": 1.0, "utmos": None, "hyp_text": "b"},
            ]
        )
        monkeypatch.setattr(
            "espnet3.systems.f5tts.metrics.versa.subprocess.Popen", scorer
        )
        metric = VersaMetric(score_config=[{"name": "signal_metric"}])

        averages = metric(self._make_data(tmp_path), "test", tmp_path / "inference")

        assert averages == {"mcd": 2.0, "utmos": 4.0}


class TestSummarize:
    """``VersaMetric.summarize`` logs a readable score table."""

    def test_logs_plain_metrics(self, caplog):
        """Plain metrics are logged under a header naming the test set."""
        with caplog.at_level("INFO", logger="espnet3.systems.f5tts.metrics.versa"):
            VersaMetric.summarize({"mcd": 1.2345}, "test")

        assert "VERSA scores - test" in caplog.text
        assert "mcd" in caplog.text

    def test_logs_wer_and_cer_component_groups(self, caplog):
        """WER and CER edit counts are grouped and reduced to one rate each."""
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

        with caplog.at_level("INFO", logger="espnet3.systems.f5tts.metrics.versa"):
            VersaMetric.summarize(scores, "test")

        assert "WER components" in caplog.text
        assert "CER components" in caplog.text
        # errors D + S + I = 4 over the reference length D + S + C = 99, not
        # over the alignment length 100: insertions are not reference words.
        assert "4.04%" in caplog.text
        assert "1.00%" in caplog.text

    def test_reports_pooled_wer_exactly_once(self, caplog):
        """The pooled scalar belongs in the WER block, not the main section too."""
        scores = {
            "fwhisper_wer_delete": 1.0,
            "fwhisper_wer_insert": 0.0,
            "fwhisper_wer_replace": 0.0,
            "fwhisper_wer_equal": 1.0,
            "fwhisper_wer": 50.0,
        }
        with caplog.at_level(
            logging.INFO, logger="espnet3.systems.f5tts.metrics.versa"
        ):
            VersaMetric.summarize(scores, test_name="unit-test")

        # "fwhisper_wer" not followed by "_": the per-op keys do not count, the
        # block header "[fwhisper_wer]:" is the one expected match.
        assert len(re.findall(r"fwhisper_wer(?!_)", caplog.text)) == 1
