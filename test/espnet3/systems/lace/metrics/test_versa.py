"""Unit tests for espnet3.systems.lace.metrics.versa."""

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

from espnet3.systems.lace.metrics.versa import VersaMetric


def _write_jsonl(path, records):
    """Write ``records`` as JSON lines, plus the blank trailer VERSA leaves."""
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


class _FakePopen:
    """Stands in for the scorer process: prints `lines`, writes `records`."""

    def __init__(self, lines=(), records=({"mcd": 3.0},), returncode=0):
        """Set the lines to print, the records to write and the exit code."""
        self._lines = list(lines)
        self._records = list(records)
        self._returncode = returncode
        self.calls = []

    def __call__(self, cmd, **kwargs):
        """Act as ``subprocess.Popen``: write the records to ``--output_file``."""
        self.calls.append(cmd)
        output_file = cmd[cmd.index("--output_file") + 1]
        _write_jsonl(pathlib.Path(output_file), self._records)
        self.stdout = io.StringIO("".join(line + "\n" for line in self._lines))
        return self

    def wait(self):
        """Return the configured exit code."""
        return self._returncode


class TestResolveScoreConfigPath:
    """``_resolve_score_config_path``: paths pass through, inline configs are dumped."""

    def test_existing_file_is_used_as_is(self, tmp_path):
        """An existing file path is returned unchanged."""
        config_file = tmp_path / "score.yaml"
        config_file.write_text("- name: signal_metric\n", encoding="utf-8")
        metric = VersaMetric(score_config=str(config_file), outputs=["mcd"])

        assert metric._resolve_score_config_path(tmp_path) == config_file

    def test_missing_file_raises(self, tmp_path):
        """A path that does not exist raises."""
        metric = VersaMetric(score_config=str(tmp_path / "nope.yaml"), outputs=["mcd"])

        with pytest.raises(FileNotFoundError):
            metric._resolve_score_config_path(tmp_path)

    def test_inline_list_is_dumped(self, tmp_path):
        """An inline list is written to ``versa_config.yaml``."""
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])

        out = metric._resolve_score_config_path(tmp_path)

        assert out == tmp_path / "versa_config.yaml"
        assert yaml.safe_load(out.read_text()) == [{"name": "signal_metric"}]

    def test_omegaconf_container_is_converted(self, tmp_path):
        """An OmegaConf container is converted before dumping."""
        metric = VersaMetric(
            score_config=OmegaConf.create([{"name": "pseudo_mos", "fs": 16000}]),
            outputs=["mcd"],
        )

        out = metric._resolve_score_config_path(tmp_path)

        assert yaml.safe_load(out.read_text()) == [{"name": "pseudo_mos", "fs": 16000}]


class TestAggregate:
    """``_aggregate``: per-utterance records become corpus-level averages."""

    def test_averages_numeric_fields_only(self, tmp_path):
        """Only numeric, non-boolean fields are averaged."""
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
        """A field missing from some rows averages over the rows that have it."""
        result_file = tmp_path / "result.json"
        _write_jsonl(result_file, [{"a": 1.0}, {"a": 2.0, "b": 10.0}])

        assert VersaMetric._aggregate(result_file) == {"a": 1.5, "b": 10.0}


class TestPooledErrorRates:
    """WER/CER pool edit-op counts over the corpus; the mean of rates is wrong."""

    def test_pools_error_counts_instead_of_averaging(self, tmp_path):
        # utt1: D=1 S=0 I=0 C=1 (50%); utt2: D=0 S=1 I=0 C=97 (1.02%).
        # Pooled: errors 2 over reference length 100 -> 2.00%, not 25.51%.
        """The pooled WER is errors over the pooled reference length."""
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
        """CER pools the same way."""
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
        """Metrics without edit-op counts are untouched."""
        result_file = tmp_path / "result.json"
        _write_jsonl(result_file, [{"utmos": 4.0}, {"utmos": 4.5}])

        scores = VersaMetric._aggregate(result_file)

        assert scores == {"utmos": 4.25}

    def test_find_prefix_requires_all_four_ops(self):
        """A prefix counts only when all four edit-op keys are present."""
        partial = {"fwhisper_wer_delete": 1, "fwhisper_wer_equal": 9}
        assert VersaMetric._find_prefix(partial, "wer") is None
        complete = _ops("fwhisper_wer", delete=1, insert=0, replace=0, equal=9)
        assert VersaMetric._find_prefix(complete, "wer") == "fwhisper_wer_"


class TestContract:
    """The metrics contract: declared inputs/outputs and the update/compute checks."""

    def test_declares_inputs_and_outputs(self):
        """``hyp`` required, ``ref``/``text`` optional; outputs follow ``outputs``."""
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["sdr"])
        assert [(f.name, f.kind, f.optional) for f in metric.inputs] == [
            ("hyp", "audio", False),
            ("ref", "audio", True),
            ("text", "text", True),
        ]
        assert [(f.name, f.kind) for f in metric.outputs] == [("sdr", "number")]

    def test_outputs_are_required(self):
        """An empty ``outputs`` list raises."""
        with pytest.raises(ValueError, match="needs `outputs`"):
            VersaMetric(score_config=[{"name": "signal_metric"}], outputs=[])

    def test_update_is_checked_against_the_declaration(self):
        """``update`` rejects a name that was not declared."""
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        with pytest.raises(Exception, match="wav"):
            metric.update(wav=["a.wav"])

    def test_compute_with_nothing_collected_raises(self):
        """``compute`` before any ``update`` raises."""
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        with pytest.raises(ValueError, match="nothing to score"):
            metric.compute()


class TestCall:
    """The full ``__call__`` path with the scorer process replaced."""

    def _make_data(self, tmp_path):
        """Write two-utterance ``hyp``/``ref`` SCPs and return the data mapping."""
        hyp = tmp_path / "resyn_audio.scp"
        ref = tmp_path / "wav_path.scp"
        hyp.write_text("utt1 a.wav\nutt2 c.wav\n", encoding="utf-8")
        ref.write_text("utt1 b.wav\nutt2 d.wav\n", encoding="utf-8")
        return {"hyp": hyp, "ref": ref}

    def test_builds_command_and_returns_averages(self, tmp_path, monkeypatch):
        """The scorer command is built and its results averaged and kept."""
        scorer = _FakePopen(lines=["scoring"], records=[{"mcd": 3.0}, {"mcd": 1.0}])
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        metric = VersaMetric(
            score_config=[{"name": "signal_metric"}], outputs=["mcd"], use_gpu=False
        )
        output_dir = tmp_path / "inference"

        # Keyword call pins the BaseMetric contract (data, test_name, output_dir).
        averages = metric(self._make_data(tmp_path), "test", output_dir=output_dir)

        assert averages == {"mcd": 2.0}
        (cmd,) = scorer.calls
        # sys.executable, not "python": under srun or a venv the interpreter
        # running the stage is the only one known to have versa installed.
        assert cmd[:3] == [sys.executable, "-m", "versa.bin.scorer"]
        assert "--use_gpu" not in cmd
        assert "--text" not in cmd
        # The collected paths were written as SCPs in collection order.
        pred = pathlib.Path(cmd[cmd.index("--pred") + 1])
        gt = pathlib.Path(cmd[cmd.index("--gt") + 1])
        eval_dir = output_dir / "test" / "scoring" / "versa_eval"
        assert json.loads((eval_dir / "avg_result.json").read_text()) == {"mcd": 2.0}
        assert (eval_dir / "result.json").is_file()
        assert (eval_dir / "versa_config.yaml").is_file()
        assert not pred.exists() and not gt.exists()  # the temp dir is gone

    def test_scp_contents_follow_collection_order(self, tmp_path, monkeypatch):
        """The SCPs list the utterances in collection order with new ids."""
        seen = {}

        class Recording(_FakePopen):
            """Capture the SCP contents before the fake scorer runs."""

            def __call__(self, cmd, **kwargs):
                """Read the ``--pred``/``--gt`` SCPs, then behave as the fake scorer."""
                seen["pred"] = pathlib.Path(cmd[cmd.index("--pred") + 1]).read_text()
                seen["gt"] = pathlib.Path(cmd[cmd.index("--gt") + 1]).read_text()
                return super().__call__(cmd, **kwargs)

        monkeypatch.setattr(
            "espnet3.systems.lace.metrics.versa.subprocess.Popen",
            Recording(records=[{"mcd": 1.0}, {"mcd": 1.0}]),
        )
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        metric(self._make_data(tmp_path), "test")
        assert seen["pred"] == "utt00000000 a.wav\nutt00000001 c.wav\n"
        assert seen["gt"] == "utt00000000 b.wav\nutt00000001 d.wav\n"

    def test_optional_inputs_may_be_absent(self, tmp_path, monkeypatch):
        """Without ``ref`` no ``--gt`` is passed."""
        scorer = _FakePopen(records=[{"mcd": 1.0}, {"mcd": 1.0}])
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        data = self._make_data(tmp_path)
        data.pop("ref")
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        metric(data, "test")
        (cmd,) = scorer.calls
        assert "--gt" not in cmd

    def test_use_gpu_and_text_are_forwarded(self, tmp_path, monkeypatch):
        """``use_gpu`` and a ``text`` input reach the command line."""
        scorer = _FakePopen(records=[{"mcd": 1.0}, {"mcd": 1.0}])
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        data = self._make_data(tmp_path)
        text = tmp_path / "text"
        text.write_text("utt1 hello\nutt2 world\n", encoding="utf-8")
        data["text"] = text
        metric = VersaMetric(
            score_config=[{"name": "signal_metric"}], outputs=["mcd"], use_gpu=True
        )
        metric(data, "test")
        (cmd,) = scorer.calls
        assert "--use_gpu" in cmd
        assert pathlib.Path(cmd[cmd.index("--text") + 1]).name == "text"

    def test_non_zero_exit_raises(self, tmp_path, monkeypatch):
        """A non-zero scorer exit raises ``CalledProcessError``."""
        scorer = _FakePopen(returncode=1)
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        with pytest.raises(subprocess.CalledProcessError):
            metric(self._make_data(tmp_path), "test")

    def test_failed_metric_is_an_error_even_when_the_scorer_exits_zero(
        self, tmp_path, monkeypatch
    ):
        """VERSA drops a metric it cannot load and still exits 0."""
        scorer = _FakePopen(
            lines=["Failed to load metric pseudo_mos: No module named 'utmos'"],
            records=[{"mcd": 3.0}, {"mcd": 1.0}],
        )
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        with pytest.raises(RuntimeError, match="Failed to load metric pseudo_mos"):
            metric(self._make_data(tmp_path), "test")

    def test_metric_that_never_scores_is_an_error(self, tmp_path, monkeypatch):
        """A metric that throws per utterance leaves its key null everywhere."""
        scorer = _FakePopen(
            records=[{"mcd": 3.0, "utmos": None}, {"mcd": 1.0, "utmos": None}]
        )
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        with pytest.raises(RuntimeError, match="no value for any utterance: utmos"):
            metric(self._make_data(tmp_path), "test")

    def test_missing_declared_output_is_an_error(self, tmp_path, monkeypatch):
        """A declared output the scorer did not emit raises."""
        scorer = _FakePopen(records=[{"mcd": 3.0}, {"mcd": 1.0}])
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        metric = VersaMetric(
            score_config=[{"name": "signal_metric"}], outputs=["mcd", "utmos"]
        )
        with pytest.raises(RuntimeError, match=r"declared outputs \['utmos'\]"):
            metric(self._make_data(tmp_path), "test")

    def test_text_fields_and_sparse_nulls_are_not_failures(self, tmp_path, monkeypatch):
        """Only a key that is null in every record and numeric in none counts."""
        scorer = _FakePopen(
            records=[
                {"mcd": 3.0, "utmos": 4.0, "hyp_text": "a"},
                {"mcd": 1.0, "utmos": None, "hyp_text": "b"},
            ]
        )
        monkeypatch.setattr("espnet3.systems.lace.metrics.versa.subprocess.Popen", scorer)
        metric = VersaMetric(score_config=[{"name": "signal_metric"}], outputs=["mcd"])
        averages = metric(self._make_data(tmp_path), "test")
        assert averages == {"mcd": 2.0, "utmos": 4.0}


class TestSummarize:
    """``summarize``: the logged table and its WER/CER blocks."""

    def test_logs_plain_metrics(self, caplog):
        """Plain metrics are logged under the test-set header."""
        with caplog.at_level("INFO", logger="espnet3.systems.lace.metrics.versa"):
            VersaMetric.summarize({"mcd": 1.2345}, "test")

        assert "VERSA scores - test" in caplog.text
        assert "mcd" in caplog.text

    def test_logs_wer_and_cer_component_groups(self, caplog):
        """Edit-op groups are logged in their own blocks with the pooled rate."""
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

        with caplog.at_level("INFO", logger="espnet3.systems.lace.metrics.versa"):
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
        with caplog.at_level(logging.INFO, logger="espnet3.systems.lace.metrics.versa"):
            VersaMetric.summarize(scores, test_name="unit-test")

        # "fwhisper_wer" not followed by "_": the per-op keys do not count, the
        # block header "[fwhisper_wer]:" is the one expected match.
        assert len(re.findall(r"fwhisper_wer(?!_)", caplog.text)) == 1
