"""Failure-analysis and debug stage for failed trials."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.interface import AgentRequest
from espnet3.bin.atlas.applications.hyperparameter_search.config_schema import (
    load_recipe_stage_config,
    write_resolved_config,
)
from espnet3.bin.atlas.applications.hyperparameter_search.stage_status import (
    _invalidate_stages,
    _read_stage_status,
)
from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.serialization import (
    apply_dotted_patch,
    save_yaml,
)
from espnet3.bin.atlas.core.stage import AtlasStage


def _glob_match(patterns, key: str) -> bool:
    """Return whether a key matches any configured glob pattern."""
    import fnmatch

    return any(fnmatch.fnmatch(key, pattern) for pattern in patterns)


class DebugTrialStage(AtlasStage):
    """Inspect failure logs and ask an agent for a patch-only retry fix."""

    _INFERENCE_RUNTIME_KEYS = frozenset(
        {"exp_dir", "stats_dir", "recipe_dir", "data_dir", "exp_tag", "inference_dir"}
    )

    def _sanitize_patch(self, context, patch: dict) -> tuple[dict, list[str]]:
        """Keep only training config keys allowed by the study search space."""
        search_space = context.config.atlas.search_space
        allowed = list(getattr(search_space, "allowed_keys", []) or [])
        denied = list(getattr(search_space, "denied_keys", []) or [])
        sanitized = {}
        removed = []
        for key, value in patch.items():
            key = str(key)
            if denied and _glob_match(denied, key):
                removed.append(key)
                continue
            if allowed and not _glob_match(allowed, key):
                removed.append(key)
                continue
            sanitized[key] = value
        return sanitized, removed

    def _sanitize_inference_patch(self, patch: dict) -> tuple[dict, list[str]]:
        """Remove inference patch keys that could redirect trial-owned outputs."""
        sanitized = {}
        removed = []
        for key, value in patch.items():
            key = str(key)
            if not key or key.split(".", 1)[0] in self._INFERENCE_RUNTIME_KEYS:
                removed.append(key)
                continue
            sanitized[key] = value
        return sanitized, removed

    def _build_request(self, context, trial) -> AgentRequest:
        trial_dir = context.trial_dir()
        logs = {
            "trial_dir": str(trial_dir),
            "training_log": str(trial_dir / "train.log"),
            "evaluation_log": str(trial_dir / "eval.log"),
            "stage_status": str(trial_dir / "stage_status.yaml"),
            "resolved_inference_config": str(
                trial_dir / "resolved_inference_config.yaml"
            ),
        }
        if trial.attempt_count > 0:
            latest_attempt = trial_dir / "attempts" / f"attempt_{trial.attempt_count:06d}"
            logs["attempt_stdout"] = str(latest_attempt / "stdout.log")
            logs["attempt_stderr"] = str(latest_attempt / "stderr.log")

        resolved_config_text = (
            (trial_dir / "resolved_training_config.yaml").read_text(encoding="utf-8")
            if (trial_dir / "resolved_training_config.yaml").exists()
            else "(missing)"
        )
        resolved_inference_text = (
            (trial_dir / "resolved_inference_config.yaml").read_text(encoding="utf-8")
            if (trial_dir / "resolved_inference_config.yaml").exists()
            else "(missing)"
        )
        stage_status_text = OmegaConf.to_yaml(
            OmegaConf.create(_read_stage_status(trial_dir))
        )

        def _load_base(config_path_key: str, config_name: str) -> str:
            try:
                from espnet3.bin.atlas.applications.hyperparameter_search.config_schema import (
                    load_recipe_stage_config,
                )
                cfg = load_recipe_stage_config(
                    context.recipe_dir,
                    Path(getattr(context.config.atlas.recipe, config_path_key)),
                    config_name,
                    resolve=False,
                )
                return OmegaConf.to_yaml(cfg)
            except Exception:
                return "(unavailable)"

        base_configs_context = "\n\n".join([
            "## Base configs\n"
            "Runtime keys are injected automatically — do NOT include them.\n",
            "### training config\n```yaml\n" + _load_base("training_config", "training.yaml") + "```",
            "### Inference config (patch it only to repair an inference-stage failure)\n```yaml\n" + _load_base("inference_config", "inference.yaml") + "```",
            "### Fixed metrics config (immutable; do not propose patches)\n```yaml\n" + _load_base("metrics_config", "metrics.yaml") + "```",
        ])

        repo_context_parts = [
            f"Trial ID: {trial.trial_id}",
            f"Trial status: {trial.status}",
            f"Current patch: {trial.config_patch}",
            "",
            "## Resolved training config (what actually ran)",
            resolved_config_text,
            "",
            "## Resolved inference config (what actually ran)",
            resolved_inference_text,
            "",
            "## Per-stage status",
            stage_status_text,
            "",
            base_configs_context,
        ]

        trial_history_csv = ""
        trials_csv = context.study_dir / "trials.csv"
        if trials_csv.exists():
            trial_history_csv = trials_csv.read_text(encoding="utf-8")

        knowledge = ""
        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        if knowledge_path.exists():
            knowledge = knowledge_path.read_text(encoding="utf-8")

        code_edit_cfg = getattr(context.config.atlas, "code_edit", None)
        allowed_dirs = list(getattr(code_edit_cfg, "allowed_dirs", ["src"]) or ["src"])
        allow_pixi = bool(getattr(code_edit_cfg, "allow_pixi", True))

        allowed_actions = ["file_write"]
        if allow_pixi:
            allowed_actions.append("pixi_add")

        return AgentRequest(
            task="debug",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=knowledge,
            repo_context="\n".join(repo_context_parts),
            trial_history_csv=trial_history_csv,
            latest_metrics=trial.metrics,
            context_files=logs,
            allowed_actions=allowed_actions,
            output_schema={
                "root_cause": "string",
                "fix_strategy": "string",
                "files_modified": ["list of src/ files edited (empty if none)"],
                "config_patches": {
                    "training": {"dotted.key": "value (or empty dict)"},
                    "inference": {"dotted.key": "value (or empty dict)"},
                },
                "should_retry": True,
            },
        )

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None
        trial_cfg = context.config.atlas.trial
        max_attempts = int(getattr(trial_cfg, "max_attempts", 0) or 0)
        retry_available = max_attempts <= 0 or trial.attempt_count < max_attempts

        request = self._build_request(context, trial)
        artifact_dir = context.trial_dir()
        response = context.agent.run(request, artifact_dir=artifact_dir)

        if response.status != "success":
            return StageResult(
                status="continue",
                message=(
                    "Debug agent failed; retrying the same trial without a fix."
                    if retry_available
                    else "Debug agent failed and the trial attempt limit was reached."
                ),
                next_node="run_trial" if retry_available else "reflect",
                payload={"trial_id": trial.trial_id},
            )

        structured = dict(response.structured or {})
        config_patches = dict(structured.get("config_patches", {}) or {})
        patch = dict(
            config_patches.get("training", {})
            or structured.get("config_patch", {})
            or {}
        )
        patch, removed = self._sanitize_patch(context, patch)
        inference_patch, removed_inference = self._sanitize_inference_patch(
            dict(config_patches.get("inference", {}) or {})
        )
        files_modified = list(structured.get("files_modified", []) or [])
        should_retry = bool(
            structured.get(
                "should_retry", bool(patch or inference_patch or files_modified)
            )
        )
        root_cause = str(structured.get("root_cause", ""))
        fix_strategy = str(structured.get("fix_strategy", response.content or ""))

        debug_note = "\n".join(
            [
                "# Debug Result",
                "",
                f"- root_cause: {root_cause}",
                f"- fix_strategy: {fix_strategy}",
                f"- files_modified: {files_modified}",
                f"- removed_training_keys: {removed}",
                f"- removed_inference_keys: {removed_inference}",
                f"- should_retry: {should_retry}",
                "",
            ]
        )
        (artifact_dir / "debug.md").write_text(debug_note, encoding="utf-8")

        if not retry_available:
            trial.reflection = (trial.reflection + "\n\n" + debug_note).strip()
            context.state.update_trial(trial)
            return StageResult(
                status="continue",
                message=f"Debug finished; {trial.trial_id} reached its attempt limit.",
                next_node="reflect",
                payload={"trial_id": trial.trial_id},
                artifacts={"debug": str(artifact_dir / "debug.md")},
            )

        if not should_retry or (not patch and not inference_patch and not files_modified):
            trial.reflection = (trial.reflection + "\n\n" + debug_note).strip()
            context.state.update_trial(trial)
            return StageResult(
                status="success",
                message="Debug finished without a fix; retrying the same trial unchanged.",
                next_node="run_trial",
                payload={"trial_id": trial.trial_id},
                artifacts={"debug": str(artifact_dir / "debug.md")},
            )

        # Both code-only and config-patch fixes retry the same trial as the next attempt.
        _runtime_keys = frozenset(
            {"exp_dir", "stats_dir", "recipe_dir", "data_dir", "exp_tag"}
        )

        trial.reflection = (trial.reflection + "\n\n" + debug_note).strip()

        inference_patch_file = artifact_dir / "inference_config_patch.yaml"
        combined_inference_patch = {}
        if inference_patch_file.exists():
            combined_inference_patch.update(
                dict(
                    OmegaConf.to_container(
                        OmegaConf.load(inference_patch_file), resolve=True
                    )
                    or {}
                )
            )
        combined_inference_patch.update(inference_patch)

        if patch:
            # Re-resolve configs in-place with the combined (parent + debug) patch.
            parent_hp_patch = {
                k: v for k, v in trial.config_patch.items() if k not in _runtime_keys
            }
            combined_hp_patch = {**parent_hp_patch, **patch}
            runtime_patch = {
                "recipe_dir": str(context.recipe_dir),
                "data_dir": str(context.recipe_dir / "data"),
                "exp_tag": trial.trial_id,
                "exp_dir": str(artifact_dir / "exp"),
            }
            combined_patch = {**combined_hp_patch, **runtime_patch}

            train_cfg = load_recipe_stage_config(
                context.recipe_dir,
                Path(context.config.atlas.recipe.training_config),
                "training.yaml",
                resolve=False,
            )
            train_unresolved = OmegaConf.to_container(train_cfg, resolve=False)
            train_unresolved = apply_dotted_patch(train_unresolved, combined_patch)
            merged_training = OmegaConf.to_container(
                OmegaConf.create(train_unresolved), resolve=True
            )
            training_out = artifact_dir / "resolved_training_config.yaml"
            write_resolved_config(training_out, OmegaConf.create(merged_training))

            # Update trial config_patch and persist
            trial.config_patch = combined_hp_patch
            save_yaml(
                artifact_dir / "config_patch.yaml", combined_hp_patch, resolve=False
            )

        if patch or inference_patch:
            runtime_only_infer = {
                "recipe_dir": str(context.recipe_dir),
                "data_dir": str(context.recipe_dir / "data"),
                "exp_tag": trial.trial_id,
                "exp_dir": str(artifact_dir / "exp"),
                "inference_dir": str(artifact_dir / "inference"),
            }
            cfg = load_recipe_stage_config(
                context.recipe_dir,
                Path(context.config.atlas.recipe.inference_config),
                "inference.yaml",
                resolve=False,
            )
            unresolved = OmegaConf.to_container(cfg, resolve=False)
            if combined_inference_patch:
                unresolved = apply_dotted_patch(unresolved, combined_inference_patch)
            unresolved = apply_dotted_patch(unresolved, runtime_only_infer)
            plain = OmegaConf.to_container(OmegaConf.create(unresolved), resolve=True)
            write_resolved_config(
                artifact_dir / "resolved_inference_config.yaml",
                OmegaConf.create(plain),
            )
            if combined_inference_patch:
                save_yaml(
                    inference_patch_file, combined_inference_patch, resolve=False
                )

        if patch:
            cfg = load_recipe_stage_config(
                context.recipe_dir,
                Path(context.config.atlas.recipe.metrics_config),
                "metrics.yaml",
                resolve=False,
            )
            unresolved = OmegaConf.to_container(cfg, resolve=False)
            unresolved = apply_dotted_patch(
                unresolved,
                {
                    "recipe_dir": str(context.recipe_dir),
                    "data_dir": str(context.recipe_dir / "data"),
                    "exp_tag": trial.trial_id,
                    "exp_dir": str(artifact_dir / "exp"),
                    "inference_dir": str(artifact_dir / "inference"),
                },
            )
            plain = OmegaConf.to_container(OmegaConf.create(unresolved), resolve=True)
            write_resolved_config(
                artifact_dir / "resolved_metrics_config.yaml",
                OmegaConf.create(plain),
            )

        if files_modified:
            stage_status = _read_stage_status(artifact_dir)
            _invalidate_stages(
                artifact_dir,
                list(stage_status.get("stages", {})),
                "source files modified",
            )
        else:
            invalidated = set()
            if patch:
                invalidated.update({"train", "infer", "measure"})
                if any(key.startswith("dataset.preprocessor.") for key in patch):
                    invalidated.add("collect_stats")
            if inference_patch:
                invalidated.update({"infer", "measure"})
            if invalidated:
                _invalidate_stages(
                    artifact_dir,
                    sorted(invalidated),
                    "debug config patch applied",
                )

        context.state.update_trial(trial)
        return StageResult(
            status="success",
            message=f"Debug fix applied to {trial.trial_id}; retrying as next attempt.",
            next_node="run_trial"
            if (patch or inference_patch) and not files_modified
            else None,
            payload={"trial_id": trial.trial_id},
            artifacts={"debug": str(artifact_dir / "debug.md")},
        )
