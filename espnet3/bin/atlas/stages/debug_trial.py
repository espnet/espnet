"""Failure-analysis and debug stage for failed trials."""

from __future__ import annotations

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import AgentRequest, StageResult
from espnet3.bin.atlas.core.utils import (
    apply_dotted_patch,
    convert_pairs_to_dict,
    save_yaml,
    split_patch_by_search_space,
)
from espnet3.bin.atlas.resources import load_prompt, load_schema
from espnet3.bin.atlas.stages.base import BaseStage
from espnet3.bin.atlas.stages.stage_status import (
    _invalidate_stages,
    _read_stage_status,
)
from espnet3.utils.config_utils import load_and_merge_config


class DebugTrialStage(BaseStage):
    """Inspect failure logs and ask an agent for a patch-only retry fix."""

    _INFERENCE_RUNTIME_KEYS = frozenset(
        {"exp_dir", "stats_dir", "recipe_dir", "data_dir", "exp_tag", "inference_dir"}
    )

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
        """Build the debug agent's request from the trial's logs, configs, and history.

        Gathers training/evaluation logs, stage status, resolved
        training/inference configs (plus the latest attempt's stdout/stderr
        if this is a retry), the recipe's base training/inference/metrics
        configs, trial history, and the knowledge pack, into one
        ``AgentRequest`` asking for a root cause, fix strategy, any files
        modified, and config patches.
        """
        trial_dir = context.get_trial_dir()
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
            latest_attempt = (
                trial_dir / "attempts" / f"attempt_{trial.attempt_count:06d}"
            )
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
            """Return the base stage config as YAML, or "(unavailable)" on error."""
            try:
                cfg = load_and_merge_config(
                    context.recipe_dir
                    / getattr(context.config.atlas.recipe, config_path_key),
                    config_name=config_name,
                    resolve=False,
                )
                return OmegaConf.to_yaml(cfg)
            except Exception:
                return "(unavailable)"

        base_configs_context = load_prompt(
            "debug_base_configs",
            training_config=_load_base("training_config", "training.yaml"),
            inference_config=_load_base("inference_config", "inference.yaml"),
            metrics_config=_load_base("metrics_config", "metrics.yaml"),
        )
        repo_context = load_prompt(
            "debug_repo_context",
            trial_id=trial.trial_id,
            status=trial.status,
            patch=trial.config_patch,
            resolved_training=resolved_config_text,
            resolved_inference=resolved_inference_text,
            stage_status=stage_status_text,
            base_configs=base_configs_context,
        )

        trial_history_csv = ""
        trials_csv = context.study_dir / "trials.csv"
        if trials_csv.exists():
            trial_history_csv = trials_csv.read_text(encoding="utf-8")

        knowledge = ""
        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        if knowledge_path.exists():
            knowledge = knowledge_path.read_text(encoding="utf-8")

        code_edit_cfg = getattr(context.config.atlas, "code_edit", None)
        allow_pixi = bool(getattr(code_edit_cfg, "allow_pixi", True))

        allowed_actions = ["file_write"]
        if allow_pixi:
            allowed_actions.append("pixi_add")

        return AgentRequest(
            task="debug",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=knowledge,
            repo_context=repo_context,
            trial_history_csv=trial_history_csv,
            latest_metrics=trial.metrics,
            context_files=logs,
            allowed_actions=allowed_actions,
            output_schema=load_schema("debug"),
        )

    def run(self, context) -> StageResult:
        """Ask the agent to diagnose a trial's failure and apply its fix (if any).

        Calls the debug agent, sanitizes any proposed training/inference
        config patches against the study's search space, re-resolves and
        writes the trial's configs when a patch is applied, invalidates
        stage-status entries affected by the fix (so they re-run), and
        decides whether to retry the trial or route to ``reflect`` (when the
        attempt limit is reached or the agent declines to retry).

        Args:
            context: The node run's ``StageContext``; must have a current
                trial.

        Returns:
            StageResult: ``status="success"`` with ``next_node="run_trial"``
            when a fix (or no-op retry) is applied, or ``status="continue"``
            routed to ``"reflect"``/``"run_trial"`` when the agent failed or
            the attempt limit was reached.

        Example:
            >>> result = DebugTrialStage().run(stage_context)
            >>> result.status
            'success'
        """
        trial = context.current_trial
        assert trial is not None
        trial_cfg = context.config.atlas.trial
        max_attempts = int(getattr(trial_cfg, "max_attempts", 0) or 0)
        retry_available = max_attempts <= 0 or trial.attempt_count < max_attempts

        request = self._build_request(context, trial)
        artifact_dir = context.get_trial_dir()
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
        config_patches = structured["config_patches"]
        patch = convert_pairs_to_dict(config_patches["training"])
        patch, removed = split_patch_by_search_space(
            patch, context.config.atlas.search_space
        )
        inference_patch, removed_inference = self._sanitize_inference_patch(
            convert_pairs_to_dict(config_patches["inference"])
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

        if not should_retry or (
            not patch and not inference_patch and not files_modified
        ):
            trial.reflection = (trial.reflection + "\n\n" + debug_note).strip()
            context.state.update_trial(trial)
            return StageResult(
                status="success",
                message="Debug found no fix; retrying the same trial unchanged.",
                next_node="run_trial",
                payload={"trial_id": trial.trial_id},
                artifacts={"debug": str(artifact_dir / "debug.md")},
            )

        # Code-only and config-patch fixes both retry the same trial.
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

            train_cfg = load_and_merge_config(
                context.recipe_dir / context.config.atlas.recipe.training_config,
                config_name="training.yaml",
                resolve=False,
            )
            merged_training = OmegaConf.to_container(
                apply_dotted_patch(train_cfg, combined_patch), resolve=True
            )
            training_out = artifact_dir / "resolved_training_config.yaml"
            save_yaml(training_out, merged_training)

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
            cfg = load_and_merge_config(
                context.recipe_dir / context.config.atlas.recipe.inference_config,
                config_name="inference.yaml",
                resolve=False,
            )
            apply_dotted_patch(cfg, combined_inference_patch)
            plain = OmegaConf.to_container(
                apply_dotted_patch(cfg, runtime_only_infer), resolve=True
            )
            save_yaml(artifact_dir / "resolved_inference_config.yaml", plain)
            if combined_inference_patch:
                save_yaml(inference_patch_file, combined_inference_patch, resolve=False)

        if patch:
            cfg = load_and_merge_config(
                context.recipe_dir / context.config.atlas.recipe.metrics_config,
                config_name="metrics.yaml",
                resolve=False,
            )
            apply_dotted_patch(
                cfg,
                {
                    "recipe_dir": str(context.recipe_dir),
                    "data_dir": str(context.recipe_dir / "data"),
                    "exp_tag": trial.trial_id,
                    "exp_dir": str(artifact_dir / "exp"),
                    "inference_dir": str(artifact_dir / "inference"),
                },
            )
            plain = OmegaConf.to_container(cfg, resolve=True)
            save_yaml(artifact_dir / "resolved_metrics_config.yaml", plain)

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
            next_node=(
                "run_trial"
                if (patch or inference_patch) and not files_modified
                else None
            ),
            payload={"trial_id": trial.trial_id},
            artifacts={"debug": str(artifact_dir / "debug.md")},
        )
