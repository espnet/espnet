"""Claude Code and Codex command-line agent."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from espnet3.bin.atlas.agents.providers import PROVIDERS, create_provider
from espnet3.bin.atlas.core.models import AgentRequest, AgentResponse
from espnet3.bin.atlas.resources import load_agent_defaults, load_prompt


def build_agent_config(config) -> DictConfig:
    """Merge ``atlas.agent`` settings over the packaged defaults and validate them.

    Args:
        config: Mapping or OmegaConf node with at least ``provider`` (one of
            :data:`PROVIDERS`, case-insensitive).

    Returns:
        DictConfig: The complete agent settings; see
        ``resources/agent_defaults.yaml`` for every key.

    Raises:
        ValueError: If ``provider`` is missing or unsupported.

    Example:
        >>> build_agent_config({"provider": "Claude"}).claude.permission_mode
        'bypassPermissions'
    """
    merged = OmegaConf.merge(load_agent_defaults(), config or {})
    provider = str(merged.provider or "").lower()
    if provider not in PROVIDERS:
        names = ", ".join(repr(name) for name in PROVIDERS)
        raise ValueError(f"atlas.agent.provider must be one of {names}")
    merged.provider = provider
    return merged


def render_prompt(request: AgentRequest) -> str:
    """Render an ``AgentRequest`` into the plain-text prompt sent to the agent.

    The JSON Schema of the reply is shown for its field descriptions; the
    provider's CLI enforces it.

    Args:
        request: The task to describe.

    Returns:
        str: The prompt text, from ``resources/prompts/agent_request.md``.
    """

    def dump(value) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    return load_prompt(
        "agent_request",
        task=request.task,
        objective=request.objective.strip(),
        knowledge=request.knowledge.strip(),
        repo_context=request.repo_context.strip(),
        trial_history_csv=request.trial_history_csv.strip(),
        latest_metrics=dump(request.latest_metrics),
        context_files=dump(request.context_files),
        inflight_patches=dump(request.inflight_patches),
        allowed_actions=", ".join(request.allowed_actions),
        output_schema=dump(request.output_schema),
    )


class CliAgent:
    """Run a coding-agent CLI as a subprocess and read its reply.

    Exposes ``run(request, artifact_dir)``. Renders the prompt, lets the
    configured provider (see ``agents/providers``) build the command and read the
    output, and keeps the artifacts of every call. A request with an
    ``output_schema`` is run with the CLI's schema-constrained final reply, and
    the reply is checked against the schema once more before it is returned.
    """

    def __init__(self, config, recipe_dir: Path) -> None:
        """Configure the agent CLI wrapper.

        Args:
            config: Mapping or OmegaConf node of ``atlas.agent``, merged over
                the defaults by :func:`build_agent_config`.
            recipe_dir: Recipe root; used as the default working directory
                when ``config.cwd`` is unset.

        Raises:
            ValueError: If ``provider`` is invalid.
        """
        self.config = build_agent_config(config)
        self.provider = create_provider(self.config)
        self.env = {str(key): str(value) for key, value in self.config.env.items()}
        self.recipe_dir = Path(recipe_dir).resolve()
        self.cwd = Path(self.config.cwd or self.recipe_dir).resolve()

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        """Execute one CLI invocation and validate its final structured response.

        Every agent call is kept on disk so a study's agent history can be
        audited afterwards. ``artifact_dir`` is the owner's directory (a trial
        dir, or the study dir for study-level skills) and ends up like this::

            <artifact_dir>/
            ├── agent_calls/
            │   └── call_<uuid>/            one per ``run()``, never overwritten
            │       ├── agent_prompt.txt    prompt sent to the CLI
            │       ├── agent_schema.json   JSON Schema the reply is constrained to
            │       ├── agent_output.txt    final reply the response is parsed from
            │       ├── agent_stdout.log    raw CLI stdout
            │       ├── agent_stderr.log    raw CLI stderr
            │       ├── agent_events.jsonl  same as stdout (JSON events, if any)
            │       └── agent_parse_error.json   only if the reply was invalid
            ├── agent_prompt.txt            latest call: the same files as in
            ├── agent_schema.json           ``call_<uuid>/``, overwritten by the
            ├── agent_output.txt            next call. The returned
            ├── agent_stdout.log            ``AgentResponse.artifacts`` points
            ├── agent_stderr.log            here, plus ``invocation`` for the
            ├── agent_events.jsonl          ``call_<uuid>`` dir.
            ├── agent_parse_error.json
            ├── agent_usage.json            latest call's token usage (codex)
            └── agent_usage_history.jsonl   usage of every call, appended (codex)

        ``agent_parse_error.json`` at the top is deleted at the start of each call,
        so it only exists when the latest call's reply was invalid.
        """
        if artifact_dir is None:
            raise ValueError("artifact_dir is required for CliAgent")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        invocation_dir = artifact_dir / "agent_calls" / f"call_{uuid.uuid4().hex}"
        invocation_dir.mkdir(parents=True)
        (artifact_dir / "agent_parse_error.json").unlink(missing_ok=True)

        def _write_artifact(path: Path, value: str) -> None:
            """Keep both the latest artifact and an immutable per-invocation copy."""
            path.write_text(value, encoding="utf-8")
            (invocation_dir / path.name).write_text(value, encoding="utf-8")

        prompt = render_prompt(request)
        prompt_path = artifact_dir / "agent_prompt.txt"
        output_path = artifact_dir / "agent_output.txt"
        stdout_path = artifact_dir / "agent_stdout.log"
        stderr_path = artifact_dir / "agent_stderr.log"
        events_path = artifact_dir / "agent_events.jsonl"
        schema_path = artifact_dir / "agent_schema.json"
        _write_artifact(prompt_path, prompt)
        if request.output_schema:
            _write_artifact(schema_path, json.dumps(request.output_schema, indent=2))
        for path in (output_path, stdout_path, stderr_path, events_path):
            _write_artifact(path, "")
        artifacts = {
            "prompt": str(prompt_path),
            "output": str(output_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "events": str(events_path),
            "invocation": str(invocation_dir),
        }

        invocation = self.provider.build_command(
            request,
            prompt,
            cwd=self.cwd,
            output_path=output_path,
            schema_path=schema_path,
        )
        command = invocation.argv
        stdin = (
            {"input": invocation.stdin}
            if invocation.stdin is not None
            else {"stdin": subprocess.DEVNULL}
        )
        try:
            proc = subprocess.run(
                command,
                **stdin,
                text=True,
                capture_output=True,
                cwd=self.cwd,
                env={
                    **os.environ,
                    **self.provider.build_env(self.recipe_dir),
                    **self.env,
                },
                timeout=self.config.timeout_sec,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                for path, value in (
                    (stdout_path, exc.stdout),
                    (stderr_path, exc.stderr),
                    (events_path, exc.stdout),
                ):
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    _write_artifact(path, value or "")
            message = (
                f"Agent executable was not found: {command[0]}"
                if isinstance(exc, FileNotFoundError)
                else f"Agent CLI timed out after {self.config.timeout_sec} seconds."
            )
            return AgentResponse(
                status="failure", message=message, content="", artifacts=artifacts
            )

        _write_artifact(stdout_path, proc.stdout)
        _write_artifact(stderr_path, proc.stderr)
        _write_artifact(events_path, proc.stdout)
        self.provider.record_usage(artifact_dir, request, command, proc.stdout)
        parsed = self.provider.parse(proc.stdout, output_path)
        content, thread_id = parsed.content, parsed.thread_id
        if proc.returncode != 0:
            return AgentResponse(
                status="failure",
                message=(
                    f"Agent CLI failed with exit code {proc.returncode}: "
                    f"{' '.join(shlex.quote(part) for part in command)}"
                ),
                content=proc.stderr.strip() or proc.stdout.strip(),
                thread_id=thread_id,
                artifacts=artifacts,
            )

        _write_artifact(output_path, content)
        if parsed.error:
            return AgentResponse(
                status="failure",
                message=parsed.error,
                content=content,
                thread_id=thread_id,
                artifacts=artifacts,
            )
        structured = {}
        if request.output_schema:
            try:
                structured = json.loads(content)
                if not isinstance(structured, dict):
                    raise ValueError(f"expected a JSON object, got {content[:80]!r}")
                self._validate_output(structured, request.output_schema)
            except ValueError as exc:
                error_path = artifact_dir / "agent_parse_error.json"
                _write_artifact(
                    error_path,
                    json.dumps({"task": request.task, "error": str(exc)}, indent=2),
                )
                artifacts["parse_error"] = str(error_path)
                return AgentResponse(
                    status="failure",
                    message=f"Invalid structured agent output: {exc}",
                    content=content,
                    thread_id=thread_id,
                    artifacts=artifacts,
                )
        return AgentResponse(
            status="success",
            message="Agent CLI completed.",
            content=content,
            structured=structured,
            thread_id=thread_id,
            artifacts=artifacts,
        )

    @staticmethod
    def _validate_output(value: dict, schema: dict) -> None:
        """Raise ``ValueError`` when ``value`` does not satisfy the JSON Schema."""
        if not schema:
            return
        from jsonschema import Draft202012Validator

        errors = sorted(
            Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.path)
        )
        if errors:
            error = errors[0]
            where = ".".join(str(part) for part in error.path) or "root"
            raise ValueError(f"{where}: {error.message}")
