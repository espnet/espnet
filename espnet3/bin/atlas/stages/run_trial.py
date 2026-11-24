"""Run one trial end-to-end."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.utils import save_yaml
from espnet3.bin.atlas.stages.base import BaseStage
from espnet3.bin.atlas.stages.stage_status import (
    _ensure_stage_statuses,
    _set_stage_status,
)


class RunTrialStage(BaseStage):
    """Execute training, inference, and measurement for one trial."""

    def _copy_training_curve_csv(self, trial_dir: Path, attempt_dir: Path) -> None:
        """Copy the latest Lightning ``metrics.csv`` to ``training_metrics.csv``."""
        csv_candidates = sorted(
            (trial_dir / "exp").glob("csv_logs/**/metrics.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        if not csv_candidates:
            return
        latest = csv_candidates[-1]
        shutil.copyfile(latest, trial_dir / "training_metrics.csv")
        shutil.copyfile(latest, attempt_dir / "training_metrics.csv")

    @staticmethod
    def _is_training_command(command: list[str]) -> bool:
        """Whether ``command`` runs the ``train`` stage (or is a custom template)."""
        try:
            return command[command.index("--stages") + 1] == "train"
        except (ValueError, IndexError):
            # A custom command template represents the complete trial workload.
            return True

    def _run_with_gpu_monitor(self, context, command, log_path, env, timeout):
        """Run training while the GPU worker performs cooperative early-stop checks."""
        interval = context.scheduler.get_gpu_monitor_interval_sec()
        started = time.monotonic()
        next_check = started + interval
        with open(log_path, "a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                command,
                cwd=str(context.recipe_dir),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while proc.poll() is None:
                now = time.monotonic()
                if timeout is not None and now - started >= timeout:
                    proc.terminate()
                    proc.wait(timeout=30)
                    return "timeout", None
                if now >= next_check:
                    try:
                        context.scheduler._maybe_check_keypoints(
                            context.current_trial.trial_id
                        )
                    except Exception as exc:  # noqa: BLE001
                        context.logger.warning(f"In-job early-stop check failed: {exc}")
                    current = context.state.get_trial(context.current_trial.trial_id)
                    if current.status == "early_stopped":
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        return "early_stopped", None
                    next_check = now + interval
                time.sleep(min(5.0, max(0.1, next_check - time.monotonic())))
        return "completed", proc

    # Stages that only need --training_config
    _TRAIN_ONLY_STAGES = frozenset(
        {"create_dataset", "train_tokenizer", "collect_stats", "train"}
    )
    # Stages that also need --inference_config
    _INFER_STAGES = frozenset({"infer"})
    # Stages that need all three configs
    _MEASURE_STAGES = frozenset({"measure"})
    # These are graph nodes which complete before RunTrialStage.  They are
    # recorded in Trial.extra_stages for routing, but are not run.py stages.
    _GRAPH_ONLY_EXTRA_STAGES = frozenset({"implement_code", "write_files"})

    def _build_stage_command(
        self,
        stage: str,
        training_cfg: Path,
        inference_cfg: Path,
        metrics_cfg: Path,
    ) -> list[str]:
        """Build the ``run.py --stages <stage>`` argv with the configs it needs."""
        cmd = [
            sys.executable,
            "run.py",
            "--stages",
            stage,
            "--training_config",
            str(training_cfg),
        ]
        if stage in self._INFER_STAGES:
            cmd += ["--inference_config", str(inference_cfg)]
        elif stage in self._MEASURE_STAGES:
            cmd += [
                "--inference_config",
                str(inference_cfg),
                "--metrics_config",
                str(metrics_cfg),
            ]
        return cmd

    def _build_default_commands(
        self,
        training_cfg: Path,
        inference_cfg: Path,
        metrics_cfg: Path,
        extra_stages: list[str] | None = None,
    ) -> list[list[str]]:
        """Build the default ``run.py`` commands (extra stages, train/infer/measure)."""
        cmds: list[list[str]] = []
        for stage in extra_stages or []:
            cmds.append(
                self._build_stage_command(
                    stage, training_cfg, inference_cfg, metrics_cfg
                )
            )
        cmds.append(
            self._build_stage_command("train", training_cfg, inference_cfg, metrics_cfg)
        )
        cmds.append(
            self._build_stage_command("infer", training_cfg, inference_cfg, metrics_cfg)
        )
        cmds.append(
            self._build_stage_command(
                "measure", training_cfg, inference_cfg, metrics_cfg
            )
        )
        return cmds

    def _build_commands(
        self, context, trial_dir: Path, attempt_dir: Path
    ) -> list[list[str]]:
        """Build this attempt's commands.

        A configured ``command_template`` wins; otherwise the default
        train/infer/measure sequence (plus any ``extra_stages``).
        """
        training_cfg = trial_dir / "resolved_training_config.yaml"
        inference_cfg = trial_dir / "resolved_inference_config.yaml"
        metrics_cfg = trial_dir / "resolved_metrics_config.yaml"
        template = list(
            getattr(context.config.atlas.trial, "command_template", []) or []
        )
        if template:
            values = {
                "trial_config": str(training_cfg),
                "training_config": str(training_cfg),
                "inference_config": str(inference_cfg),
                "metrics_config": str(metrics_cfg),
                "trial_dir": str(trial_dir),
                "attempt_dir": str(attempt_dir),
                "recipe_dir": str(context.recipe_dir),
            }
            return [[str(part).format(**values) for part in template]]

        trial = context.current_trial
        extra_stages = [
            stage
            for stage in (getattr(trial, "extra_stages", None) or [])
            if stage not in self._GRAPH_ONLY_EXTRA_STAGES
        ]
        return self._build_default_commands(
            training_cfg, inference_cfg, metrics_cfg, extra_stages
        )

    @staticmethod
    def _get_command_stage(command: list[str]) -> str:
        """Return the recipe stage named by a command, or its custom-workload name."""
        try:
            return command[command.index("--stages") + 1]
        except (ValueError, IndexError):
            return "custom_command"

    @staticmethod
    def _is_training_stage(stage_name: str) -> bool:
        """Return whether a stage belongs in the training log."""
        return stage_name in RunTrialStage._TRAIN_ONLY_STAGES

    @staticmethod
    def _prepare_resume_config(
        training_cfg: Path, attempt_dir: Path, checkpoint_path: Path
    ) -> Path | None:
        """Create an attempt config that resumes from an existing last checkpoint."""
        if not checkpoint_path.is_file():
            return None

        config = OmegaConf.load(training_cfg)
        OmegaConf.update(
            config,
            "fit.ckpt_path",
            str(checkpoint_path.resolve()),
            merge=True,
            force_add=True,
        )
        resume_cfg = attempt_dir / "resolved_training_config.yaml"
        OmegaConf.save(config, resume_cfg)
        return resume_cfg

    def run(self, context) -> StageResult:
        """Run a trial's train/infer/measure commands as a local subprocess pipeline.

        Skips stages already marked ``"completed"`` in ``stage_status.yaml``
        (so a retried attempt resumes rather than redoing finished work),
        resumes training from ``exp/last.ckpt`` when present, writes a
        ``train_eval.sh`` script recording the commands run, and — when
        ``context.gpu_self_controller`` is set — runs the training command
        under cooperative early-stop monitoring instead of a plain blocking
        call.

        Args:
            context: The node run's ``StageContext``; must have a current
                trial.

        Returns:
            StageResult: ``status="success"`` if every command exits 0,
            ``status="failure"`` if a command exits non-zero,
            ``status="timeout"`` if a command exceeds
            ``trial.timeout_sec``, or ``status="early_stopped"`` if the
            keypoint agent stopped the trial mid-training (GPU
            self-controller mode only).

        Example:
            >>> result = RunTrialStage().run(stage_context)
            >>> result.status
            'success'
        """
        trial = context.current_trial
        assert trial is not None
        trial_dir = context.get_trial_dir()
        attempt_dir = trial_dir / "attempts" / context.attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        training_cfg = trial_dir / "resolved_training_config.yaml"
        train_log = trial_dir / "train.log"
        eval_log = trial_dir / "eval.log"
        commands = self._build_commands(context, trial_dir, attempt_dir)
        stage_names = [self._get_command_stage(command) for command in commands]
        stage_status = _ensure_stage_statuses(trial_dir, stage_names)
        previous_stage_states = {
            name: entry.get("previous_status") or entry.get("status")
            for name, entry in stage_status["stages"].items()
        }
        completed_stages = {
            stage_name
            for stage_name, entry in stage_status["stages"].items()
            if entry.get("status") == "completed"
        }
        commands = [
            command
            for command in commands
            if self._get_command_stage(command) not in completed_stages
        ]
        if len(commands) < len(stage_names):
            context.logger.info(
                "Skipping completed stages for %s: %s",
                trial.trial_id,
                sorted(completed_stages.intersection(stage_names)),
            )
        resume_cfg = None
        if any(self._is_training_command(command) for command in commands):
            resume_cfg = self._prepare_resume_config(
                training_cfg,
                attempt_dir,
                trial_dir / "exp" / "last.ckpt",
            )
        if resume_cfg is not None:
            for command in commands:
                command[:] = [
                    str(resume_cfg) if part == str(training_cfg) else part
                    for part in command
                ]
            context.logger.info(
                "Resuming %s from checkpoint %s",
                trial.trial_id,
                trial_dir / "exp" / "last.ckpt",
            )

        script_lines = [
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            f"cd {context.recipe_dir}",
            *[
                " ".join(shlex.quote(str(part)) for part in command)
                for command in commands
            ],
            "",
        ]
        script_path = trial_dir / "train_eval.sh"
        script_path.write_text("\n".join(script_lines), encoding="utf-8")
        script_path.chmod(0o755)

        trial.status = "running"
        trial.attempt_count += 1
        context.state.update_trial(trial)
        timeout = getattr(context.config.atlas.trial, "timeout_sec", None)
        save_yaml(
            attempt_dir / "job.yaml",
            {
                "trial_id": trial.trial_id,
                "attempt_id": context.attempt_id,
                "commands": commands,
            },
        )
        env = os.environ.copy()
        current_stage = None
        try:
            proc = None
            for command in commands:
                current_stage = self._get_command_stage(command)
                if current_stage == "infer" and previous_stage_states.get(
                    current_stage
                ) in {"failed", "timeout"}:
                    inference_dir = trial_dir / "inference"
                    stale_locks = list(inference_dir.glob("**/split.*/lock"))
                    for lock_path in stale_locks:
                        lock_path.unlink(missing_ok=True)
                    if stale_locks:
                        context.logger.warning(
                            "Removed %d stale inference shard lock(s) before "
                            "retrying %s",
                            len(stale_locks),
                            trial.trial_id,
                        )
                log_path = (
                    train_log
                    if self._is_training_stage(current_stage)
                    or current_stage == "custom_command"
                    else eval_log
                )
                _set_stage_status(
                    trial_dir, current_stage, "running", context.attempt_id
                )
                if context.gpu_self_controller and self._is_training_command(command):
                    monitor_status, proc = self._run_with_gpu_monitor(
                        context, command, log_path, env, timeout
                    )
                    if monitor_status == "early_stopped":
                        _set_stage_status(
                            trial_dir,
                            current_stage,
                            "early_stopped",
                            context.attempt_id,
                        )
                        return StageResult(
                            status="early_stopped", message="Trial early stopped"
                        )
                    if monitor_status == "timeout":
                        _set_stage_status(
                            trial_dir,
                            current_stage,
                            "timeout",
                            context.attempt_id,
                        )
                        trial.status = "timeout"
                        context.state.update_trial(trial)
                        return StageResult(status="timeout", message="Trial timed out")
                else:
                    with open(log_path, "a", encoding="utf-8") as handle:
                        proc = subprocess.run(
                            command,
                            cwd=str(context.recipe_dir),
                            env=env,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=timeout,
                            check=False,
                        )
                if proc.returncode != 0:
                    _set_stage_status(
                        trial_dir,
                        current_stage,
                        "failed",
                        context.attempt_id,
                        return_code=proc.returncode,
                    )
                    trial.status = "failed"
                    context.state.update_trial(trial)
                    return StageResult(
                        status="failure",
                        message=(
                            f"Stage '{current_stage}' failed with exit code "
                            f"{proc.returncode}: {' '.join(command)}"
                        ),
                    )
                _set_stage_status(
                    trial_dir,
                    current_stage,
                    "completed",
                    context.attempt_id,
                    return_code=proc.returncode,
                )
        except subprocess.TimeoutExpired:
            if current_stage is not None:
                _set_stage_status(
                    trial_dir, current_stage, "timeout", context.attempt_id
                )
            trial.status = "timeout"
            context.state.update_trial(trial)
            return StageResult(status="timeout", message="Trial timed out")

        trial.status = "completed"
        context.state.update_trial(trial)
        self._copy_training_curve_csv(trial_dir, attempt_dir)
        metrics_path = trial_dir / "inference" / "metrics.json"
        if metrics_path.exists():
            target_metrics = trial_dir / "metrics.json"
            target_metrics.write_text(
                metrics_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return StageResult(
            status="success",
            message="Trial completed",
            artifacts={
                "train_log": str(train_log),
                "eval_log": str(eval_log),
                "training_metrics_csv": str(trial_dir / "training_metrics.csv"),
                "metrics": str(trial_dir / "metrics.json"),
            },
        )
