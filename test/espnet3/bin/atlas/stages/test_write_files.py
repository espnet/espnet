"""Tests for espnet3.bin.atlas.stages.write_files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import AgentResponse, Trial
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages import write_files
from espnet3.bin.atlas.stages.write_files import WriteFilesStage

STUDY = "study"


def load_yaml(path):
    """Load a YAML file as plain containers."""
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _trial(trial_id="trial_000001", score=None, score_name="wer", **kwargs):
    """Build a minimal trial."""
    now = datetime.now(timezone.utc).isoformat()
    fields = dict(
        trial_id=trial_id,
        study_id=STUDY,
        status="proposed",
        config_patch={},
        resolved_config_path=None,
        rationale="",
        expected_effect="",
        risk="",
        score=score,
        score_name=score_name,
        decision=None,
        parent_trial_id=None,
        attempt_count=0,
        created_at=now,
        updated_at=now,
    )
    fields.update(kwargs)
    return Trial(**fields)


def _context(tmp_path, *, trial=None, atlas=None, agent=None):
    """Build a StageContext stand-in backed by a real StateStore."""
    study_dir = tmp_path / "study"
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir(exist_ok=True)
    state = StateStore(study_dir)
    state.create_or_load_study(STUDY, study_dir)
    if trial is not None:
        state.create_trial(trial)
    trial_dir = study_dir / "trials" / (trial.trial_id if trial else "none")
    trial_dir.mkdir(parents=True, exist_ok=True)
    base = {"metric": {"name": "wer", "mode": "min", "extraction": []}}
    base.update(atlas or {})
    return SimpleNamespace(
        config=OmegaConf.create({"atlas": base}),
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        current_trial=trial,
        state=state,
        agent=agent,
        scheduler=SimpleNamespace(study_id=STUDY),
        logger=logging.getLogger("test_atlas"),
        get_trial_dir=lambda: trial_dir,
        get_artifact_dir=lambda: study_dir / "node_runs" / "run",
    )


def test_parse_file_blocks_extracts_paths_and_contents_in_order():
    text = (
        'intro\n<file path="src/a.py">\nx = 1\n</file>\n'
        'mid <file path="tests/test_a.py">def test(): pass</file> end'
    )
    assert write_files.parse_file_blocks(text) == [
        ("src/a.py", "x = 1"),
        ("tests/test_a.py", "def test(): pass"),
    ]
    assert write_files.parse_file_blocks("no blocks") == []


def test_is_allowed_accepts_only_paths_inside_allowed_dirs():
    allowed = ["src", "tests/unit"]
    assert write_files._is_allowed("src/m.py", allowed)
    assert write_files._is_allowed("src/deep/m.py", allowed)
    assert write_files._is_allowed("tests/unit/t.py", allowed)
    assert write_files._is_allowed("src", allowed)
    assert not write_files._is_allowed("other/m.py", allowed)
    assert not write_files._is_allowed("srcx/m.py", allowed)
    assert not write_files._is_allowed("tests/t.py", allowed)


def test_read_files_renders_blocks_and_skips_pycache_and_binary(tmp_path):
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "__pycache__" / "m.pyc").write_text("x")
    (tmp_path / "src" / "m.py").write_text("code", encoding="utf-8")
    (tmp_path / "src" / "b.bin").write_bytes(b"\xff\xfe\x00")

    text = write_files._read_files(tmp_path, ["src", "absent"])

    assert '<file path="src/b.bin">\n(binary or unreadable)\n</file>' in text
    assert '<file path="src/m.py">\ncode\n</file>' in text
    assert "pyc" not in text


def test_read_reference_files_reports_missing_and_existing(tmp_path):
    ref = tmp_path / "ref.py"
    ref.write_text("print(1)", encoding="utf-8")
    text = write_files._read_reference_files([str(ref), str(tmp_path / "no.py")])
    assert "# ref.py\n```python\nprint(1)\n```" in text
    assert "(file not found)" in text


def test_read_reference_files_reports_unreadable_files(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_bytes(b"\xff\xfe")
    assert "(unreadable:" in write_files._read_reference_files([str(bad)])


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_fetch_url_strips_html_tags(monkeypatch):
    import urllib.request

    body = b"<html><b>Hello</b>   world\n\n\n\n<i>!</i></html>"
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: _FakeResponse(body)
    )
    text = write_files._fetch_url("http://x")
    assert "Hello" in text and "<" not in text and "\n\n\n" not in text


def test_fetch_url_reports_failures_and_empty_pages(monkeypatch):
    import urllib.request

    def boom(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert write_files._fetch_url("http://x").startswith("(failed to fetch http://x")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: _FakeResponse(b"<p></p>")
    )
    assert write_files._fetch_url("http://y") == "(empty response from http://y)"


def test_fetch_doc_urls_caches_each_page(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return f"content of {url}"

    monkeypatch.setattr(write_files, "_fetch_url", fake_fetch)
    cache = tmp_path / "cache"

    first = write_files._fetch_doc_urls(["http://a/x", "http://b/y"], cache)
    second = write_files._fetch_doc_urls(["http://a/x", "http://b/y"], cache)

    assert fetched == ["http://a/x", "http://b/y"]
    assert first == second
    assert "## Doc: http://a/x\n\ncontent of http://a/x" in first
    assert "\n\n---\n\n" in first
    assert len(list(cache.iterdir())) == 2


def test_read_training_config_and_find_latest_failure(tmp_path):
    assert "not found" in write_files._read_training_config(tmp_path)
    (tmp_path / "resolved_training_config.yaml").write_text("d: 4\n")
    assert write_files._read_training_config(tmp_path) == "d: 4\n"

    assert write_files._find_latest_test_failure(tmp_path) is None
    runs = tmp_path / "test_runs"
    for name, err in [("run_001", "old"), ("run_002", None), ("run_003", "new")]:
        (runs / name).mkdir(parents=True)
        if err:
            (runs / name / "error.log").write_text(err, encoding="utf-8")
    assert write_files._find_latest_test_failure(tmp_path) == "new"
    (runs / "run_003" / "error.log").unlink()
    assert write_files._find_latest_test_failure(tmp_path) == "old"
    (runs / "run_001" / "error.log").unlink()
    assert write_files._find_latest_test_failure(tmp_path) is None


def test_create_next_iteration_dir_numbers_iterations(tmp_path):
    first = write_files._create_next_iteration_dir(tmp_path)
    second = write_files._create_next_iteration_dir(tmp_path)
    assert (first.name, second.name) == ("iter_001", "iter_002")


def test_build_prompt_includes_context_and_fix_instructions(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1", encoding="utf-8")
    trial = _trial(rationale="Use wider layers\nmore detail")

    fix = write_files._build_prompt(
        trial, tmp_path, ["src"], "REFS", "DOCS", "d_model: 4", "AssertionError!"
    )

    assert "Rationale: Use wider layers\n" in fix
    assert "more detail" not in fix
    for part in ["FAILED unit tests", "AssertionError!", "d_model: 4", "REFS"]:
        assert part in fix
    assert "DOCS" in fix and '<file path="src/m.py">' in fix
    assert "This is a fix attempt" in fix

    plain = write_files._build_prompt(_trial(), tmp_path, ["empty"], "", "", "", None)
    assert "Rationale: (none)" in plain
    assert "(no files yet)" in plain
    assert "FAILED" not in plain and "fix attempt" not in plain


def _code_edit(**kwargs):
    return {"code_edit": kwargs}


class _ScriptedAgent:
    """Return a fixed response and remember the request."""

    def __init__(self, response):
        self.response = response
        self.request = None
        self.artifact_dir = None

    def run(self, request, artifact_dir=None):
        self.request = request
        self.artifact_dir = artifact_dir
        return self.response


def _ok(content):
    return AgentResponse(status="success", message="ok", content=content)


def _write_context(tmp_path, agent, **code_edit):
    context = _context(
        tmp_path, trial=_trial(), atlas=_code_edit(**code_edit), agent=agent
    )
    (context.study_dir / "program.md").write_text("objective", encoding="utf-8")
    return context


def test_write_files_without_allowed_dirs_is_a_no_op(tmp_path):
    agent = _ScriptedAgent(_ok(""))
    context = _write_context(tmp_path, agent)

    result = WriteFilesStage().run(context)

    assert result.status == "success"
    assert "not configured" in result.message
    assert agent.request is None


def test_write_files_applies_allowed_blocks_and_backs_up_overwritten_files(tmp_path):
    output = (
        '<file path="src/model.py">\nNEW\n</file>\n'
        '<file path="src/new.py">\nfresh\n</file>\n'
        '<file path="evil/x.py">\nbad\n</file>\nsummary'
    )
    agent = _ScriptedAgent(_ok(output))
    context = _write_context(tmp_path, agent, allowed_dirs=["src"])
    (context.recipe_dir / "src").mkdir()
    (context.recipe_dir / "src" / "model.py").write_text("OLD", encoding="utf-8")
    (context.study_dir / "knowledge").mkdir()
    (context.study_dir / "knowledge" / "knowledge_pack.md").write_text("KNOW")

    result = WriteFilesStage().run(context)

    assert result.status == "success"
    assert result.payload["written"] == ["src/model.py", "src/new.py"]
    assert (context.recipe_dir / "src" / "model.py").read_text() == "NEW"
    assert (context.recipe_dir / "src" / "new.py").read_text() == "fresh"
    assert not (context.recipe_dir / "evil").exists()
    iter_dir = Path(result.artifacts["iter_dir"])
    assert (iter_dir / "backup" / "src" / "model.py").read_text() == "OLD"
    assert not (iter_dir / "backup" / "src" / "new.py").exists()
    assert (iter_dir / "agent_output.txt").read_text() == output
    assert "OLD" in (iter_dir / "prompt.txt").read_text()
    meta = load_yaml(iter_dir / "result.yaml")
    assert meta["skipped_not_allowed"] == ["evil/x.py"]
    assert meta["is_fix_attempt"] is False
    assert agent.request.task == "write_files"
    assert agent.request.objective == "objective"
    assert agent.request.knowledge == "KNOW"
    assert agent.artifact_dir == iter_dir


def test_write_files_prompt_carries_reference_docs_and_previous_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(write_files, "_fetch_url", lambda url: "DOC BODY")
    ref = tmp_path / "ref.py"
    ref.write_text("REFERENCE", encoding="utf-8")
    agent = _ScriptedAgent(_ok("nothing to write"))
    context = _write_context(
        tmp_path,
        agent,
        allowed_dirs=["src"],
        reference_files=[str(ref)],
        doc_urls=["http://docs/page"],
    )
    failed = context.get_trial_dir() / "test_runs" / "run_001"
    failed.mkdir(parents=True)
    (failed / "error.log").write_text("KABOOM", encoding="utf-8")

    result = WriteFilesStage().run(context)

    assert result.status == "success"
    assert "no <file> blocks" in result.message
    prompt = agent.request.repo_context
    assert "REFERENCE" in prompt and "DOC BODY" in prompt and "KABOOM" in prompt
    assert (context.study_dir / ".doc_cache").is_dir()


def test_write_files_fails_when_the_agent_fails(tmp_path):
    agent = _ScriptedAgent(AgentResponse(status="failure", message="down", content=""))
    context = _write_context(tmp_path, agent, allowed_dirs=["src"])

    result = WriteFilesStage().run(context)

    assert result.status == "failure"
    assert "down" in result.message
