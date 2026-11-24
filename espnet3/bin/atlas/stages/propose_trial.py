"""Propose the next hyperparameter trial."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import AgentRequest, StageResult, Trial
from espnet3.bin.atlas.core.utils import (
    apply_dotted_patch,
    convert_pairs_to_dict,
    get_utc_now,
    save_yaml,
    split_patch_by_search_space,
)
from espnet3.bin.atlas.resources import load_schema
from espnet3.bin.atlas.stages.base import BaseStage
from espnet3.utils.config_utils import load_and_merge_config

ALLOWED_ACTIONS_BY_APPLICATION = {
    "hyperparameter_search": ["config_patch_only"],
    "architecture_search": ["config_patch", "implement_code", "pixi_add"],
}


def _normalize_trial_config(value, trial_dir: Path):
    """Remove trial-specific paths before comparing effective configs."""
    if isinstance(value, dict):
        return {
            key: _normalize_trial_config(item, trial_dir)
            for key, item in value.items()
            if key not in {"exp_dir", "exp_tag", "recipe_dir", "data_dir"}
        }
    if isinstance(value, list):
        return [_normalize_trial_config(item, trial_dir) for item in value]
    if isinstance(value, str):
        return value.replace(str(trial_dir), "<trial_dir>")
    return value


def _compute_config_signature(config, trial_dir: Path) -> str:
    """Create a stable signature for a resolved training config."""
    plain = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    normalized = _normalize_trial_config(plain, trial_dir)
    # Logger names identify runs; they do not distinguish training experiments.
    # In particular, a trial-specific Wandb name must not defeat deduplication.
    if isinstance(normalized.get("trainer"), dict):
        normalized["trainer"].pop("logger", None)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)


class ProposeTrialStage(BaseStage):
    """Create a new Trial from the agent's proposed config patch."""

    study_lock_name = "study_controller"

    def _compute_next_trial_id(self, trials) -> str:
        """Return the next sequential ``trial_NNNNNN`` id after ``trials``."""
        return f"trial_{len(trials) + 1:06d}"

    def _ensure_provisional_trial(self, context, latest_trial, trials) -> Trial:
        """Reserve a trial, retaining its ID when JSON generation needs repair."""
        if self._should_reuse_current_trial(context, latest_trial):
            return latest_trial
        trial_id = self._compute_next_trial_id(trials)
        trial_dir = context.study_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial = Trial(
            trial_id=trial_id,
            study_id=str(context.scheduler.study_id),
            status="proposing",
            config_patch={},
            resolved_config_path=None,
            rationale="",
            expected_effect="",
            risk="",
            score=None,
            score_name=str(context.config.atlas.metric.name),
            decision=None,
            parent_trial_id=latest_trial.trial_id if latest_trial else None,
            attempt_count=0,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
        )
        context.state.create_trial(trial)
        save_yaml(
            trial_dir / "trial.yaml",
            {"trial_id": trial_id, "status": trial.status, "pending_agent": None},
        )
        return trial

    def _should_reuse_current_trial(self, context, current_trial: Trial | None) -> bool:
        """Whether a ``"proposing"`` trial with no resolved config yet should be reused.

        Lets a retried ``propose_trial`` node run (e.g. after an agent
        repair) continue the same provisional trial id instead of creating
        a new one.
        """
        if current_trial is None:
            return False
        if current_trial.status != "proposing":
            return False
        if current_trial.resolved_config_path:
            return False
        return True

    def _build_runtime_path_patch(
        self, context, trial_dir: Path, trial_id: str
    ) -> dict:
        """Build the runtime-only patch keys (paths, exp_tag) given to every trial."""
        return {
            "recipe_dir": str(context.recipe_dir),
            "data_dir": str(context.recipe_dir / "data"),
            "exp_tag": trial_id,
            "exp_dir": str(trial_dir / "exp"),
        }

    def run(self, context) -> StageResult:
        """Ask the agent for a new config patch, resolve it, and save it as a trial.

        Reserves (or reuses) a provisional trial, asks the agent to propose
        a training config patch, sanitizes the patch against the search space,
        rejects it as ``"duplicate"`` if an in-progress or past trial already
        has the same hyperparameter patch
        or resolves to the same effective training config, then writes the
        trial's resolved training/inference/metrics configs and metadata.

        Args:
            context: The node run's ``StageContext``.

        Returns:
            StageResult: ``status="success"`` (or ``"write_files"``/
            ``"implement_code"`` for an architecture-search trial needing
            extra stages first) once a trial is proposed;
            ``status="no_budget"`` if the study's budget is exhausted;
            ``status="duplicate"``/``"failure"`` if no valid proposal could
            be produced (including when the agent call itself fails).

        Example:
            >>> result = ProposeTrialStage().run(stage_context)
            >>> result.status
            'success'
        """
        if context.scheduler.should_stop():
            return StageResult(status="no_budget", message="Budget exhausted")

        trials = context.state.list_trials(context.scheduler.study_id)
        latest_trial = trials[-1] if trials else None
        current_trial = context.current_trial

        _runtime_keys = frozenset(
            {"exp_dir", "stats_dir", "recipe_dir", "data_dir", "exp_tag"}
        )

        if self._should_reuse_current_trial(context, current_trial):
            trial = current_trial
        else:
            trial = self._ensure_provisional_trial(context, latest_trial, trials)

        trial_dir = context.study_dir / "trials" / trial.trial_id
        trial_id = trial.trial_id

        # Build inflight patches from trials currently in flight
        inflight_patches = []
        for t in trials:
            if (
                t.status in frozenset({"running", "proposing", "proposed"})
                and t.trial_id != trial_id
            ):
                patch = {
                    k: v for k, v in t.config_patch.items() if k not in _runtime_keys
                }
                inflight_patches.append(patch)

        # Knowledge and history
        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        knowledge = (
            knowledge_path.read_text(encoding="utf-8")
            if knowledge_path.exists()
            else ""
        )
        checklist_path = context.recipe_dir / "src-checklist.md"
        if checklist_path.is_file():
            knowledge += (
                "\n\n## Required persistent checklist (read and update after a trial)\n"
                f"Path: {checklist_path}\n\n"
                + checklist_path.read_text(encoding="utf-8")
            )
        trials_csv = context.study_dir / "trials.csv"
        trial_history_csv = (
            trials_csv.read_text(encoding="utf-8") if trials_csv.exists() else ""
        )

        # The skill reads logs on demand instead of copying them into every prompt.
        context_files = {
            "study_dir": str(context.study_dir),
            "trial_dir": str(trial_dir),
            "training_log": str(trial_dir / "train.log"),
            "evaluation_log": str(trial_dir / "eval.log"),
            "training_metrics_csv": str(trial_dir / "training_metrics.csv"),
            "all_trials_dir": str(context.study_dir / "trials"),
            "checklist": str(checklist_path),
        }

        # Load all base configs so the agent can see what is patchable
        def _load_base_config_text(config_path_key: str, config_name: str) -> str:
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

        base_configs_context = "\n\n".join(
            [
                "## Base configs\n"
                "Runtime keys (exp_dir, stats_dir, recipe_dir, data_dir, exp_tag, "
                "inference_dir) "
                "are injected automatically — do NOT include them in your patches.\n",
                "Locked training keys: trainer.max_epochs, dataloader.*, "
                "dataset.train, and dataset.valid. Do not propose patches for them.\n",
                "### training config\n```yaml\n"
                + _load_base_config_text("training_config", "training.yaml")
                + "```",
                "### Fixed inference config (immutable; do not propose patches)\n"
                "```yaml\n"
                + _load_base_config_text("inference_config", "inference.yaml")
                + "```",
                "### Fixed metrics config (immutable; do not propose patches)\n"
                "```yaml\n"
                + _load_base_config_text("metrics_config", "metrics.yaml")
                + "```",
            ]
        )

        repo_context = base_configs_context

        application = str(context.config.atlas.application)
        allowed_actions = ALLOWED_ACTIONS_BY_APPLICATION.get(
            application, ["config_patch_only"]
        )

        if "implement_code" in allowed_actions:
            extra_stages_hint = (
                "list of stage names to run before train — "
                "include 'implement_code' to write/modify src/ files for a new "
                "architecture, "
                "or 'train_tokenizer' if a new tokenizer is needed"
            )
        else:
            extra_stages_hint = (
                "optional list of stage names to run before train, e.g. train_tokenizer"
            )

        output_schema = load_schema("propose")
        output_schema["properties"]["extra_stages"]["description"] = extra_stages_hint

        request = AgentRequest(
            task="propose",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=knowledge,
            repo_context=repo_context,
            trial_history_csv=trial_history_csv,
            latest_metrics={},
            context_files=context_files,
            inflight_patches=inflight_patches,
            allowed_actions=allowed_actions,
            output_schema=output_schema,
        )
        artifact_dir = trial_dir
        response = context.agent.run(request, artifact_dir=artifact_dir)

        if response.status != "success":
            trial.status = "cancelled"
            context.state.update_trial(trial)
            return StageResult(
                status="failure",
                message=f"Proposal agent failed: {response.message}",
                payload={"trial_id": trial_id},
            )

        structured = dict(response.structured or {})
        patch = convert_pairs_to_dict(structured["config_patches"]["training"])
        rationale = str(structured.get("rationale", response.content or ""))
        expected_effect = str(structured.get("expected_effect", ""))
        risk = str(structured.get("risk", ""))
        _known_stages = {
            "create_dataset",
            "train_tokenizer",
            "collect_stats",
            "implement_code",
            "write_files",
        }
        extra_stages = [
            str(s)
            for s in (structured.get("extra_stages") or [])
            if str(s) in _known_stages
        ]

        patch, removed = split_patch_by_search_space(
            patch, context.config.atlas.search_space
        )
        if not patch and removed:
            trial.status = "cancelled"
            context.state.update_trial(trial)
            return StageResult(
                status="failure",
                message=f"Patch became empty after validation. removed={removed}",
                payload={"trial_id": trial_id},
            )

        # Deduplicate
        hp_patch = {k: v for k, v in patch.items() if k not in _runtime_keys}
        for existing in trials:
            existing_hp = {
                k: v for k, v in existing.config_patch.items() if k not in _runtime_keys
            }
            if existing_hp == hp_patch and existing.trial_id != trial_id:
                dup_id = existing.trial_id
                trial.status = "cancelled"
                context.state.update_trial(trial)
                return StageResult(
                    status="duplicate",
                    message=f"Duplicate patch: same as {dup_id}",
                    payload={"trial_id": trial_id},
                )

        # Build runtime path patch and merge
        runtime_patch = self._build_runtime_path_patch(context, trial_dir, trial_id)
        effective_patch = {**patch, **runtime_patch}

        # Load training config without resolving so that OmegaConf interpolations
        # (e.g. model.vocab_size: ${tokenizer.vocab_size}) remain as strings.
        # Applying the patch before resolution ensures dependent fields propagate
        # correctly when tokenizer settings change.
        train_cfg = load_and_merge_config(
            context.recipe_dir / context.config.atlas.recipe.training_config,
            config_name="training.yaml",
            resolve=False,
        )
        train_plain = OmegaConf.to_container(
            apply_dotted_patch(train_cfg, effective_patch), resolve=True
        )
        # Sparse patches can differ while resolving to the same effective
        # training configuration (for example, omitting a key at its default
        # value). Compare resolved configs too, ignoring generated trial paths.
        candidate_signature = _compute_config_signature(train_plain, trial_dir)
        for existing in trials:
            existing_path = existing.resolved_config_path
            if (
                existing.trial_id == trial_id
                or existing_path is None
                or not existing_path.is_file()
            ):
                continue
            existing_config = OmegaConf.load(existing_path)
            existing_dir = context.study_dir / "trials" / existing.trial_id
            existing_signature = _compute_config_signature(
                existing_config, existing_dir
            )
            if existing_signature == candidate_signature:
                trial.status = "cancelled"
                context.state.update_trial(trial)
                return StageResult(
                    status="duplicate",
                    message=(
                        "Duplicate effective training config: same as "
                        f"{existing.trial_id}"
                    ),
                    payload={"trial_id": trial_id, "duplicate_of": existing.trial_id},
                )

        training_out = trial_dir / "resolved_training_config.yaml"
        save_yaml(training_out, train_plain)

        # Evaluation is a study invariant.  Never apply either training HPs or
        # agent-provided config_patches.inference/metrics to these configs; only
        # trial-local runtime paths are substituted below.
        runtime_inference_paths = {
            **runtime_patch,
            "inference_dir": str(trial_dir / "inference"),
        }
        for config_key, config_name, out_name in [
            ("inference_config", "inference.yaml", "resolved_inference_config.yaml"),
            ("metrics_config", "metrics.yaml", "resolved_metrics_config.yaml"),
        ]:
            cfg = load_and_merge_config(
                context.recipe_dir / getattr(context.config.atlas.recipe, config_key),
                config_name=config_name,
                resolve=False,
            )
            plain = OmegaConf.to_container(
                apply_dotted_patch(cfg, runtime_inference_paths), resolve=True
            )
            save_yaml(trial_dir / out_name, plain)

        inference_out = trial_dir / "resolved_inference_config.yaml"
        metrics_out = trial_dir / "resolved_metrics_config.yaml"

        # Save metadata
        hp_patch_only = {
            k: v for k, v in effective_patch.items() if k not in _runtime_keys
        }
        save_yaml(trial_dir / "config_patch.yaml", hp_patch_only, resolve=False)
        (trial_dir / "rationale.md").write_text(rationale, encoding="utf-8")

        # Update trial
        trial.status = "proposed"
        trial.config_patch = effective_patch
        trial.resolved_config_path = training_out
        trial.rationale = rationale
        trial.expected_effect = expected_effect
        trial.risk = risk
        trial.updated_at = get_utc_now()
        if response.status == "success" and getattr(response, "thread_id", None):
            trial.codex_thread_id = response.thread_id
        trial.extra_stages = extra_stages
        context.state.update_trial(trial)
        save_yaml(
            trial_dir / "trial.yaml",
            {
                "trial_id": trial_id,
                "status": trial.status,
                "config_patch": hp_patch_only,
                "rationale": rationale,
                "expected_effect": expected_effect,
                "risk": risk,
                "removed_keys": removed,
                "extra_stages": extra_stages,
                "codex_thread_id": trial.codex_thread_id,
            },
            resolve=False,
        )

        if "write_files" in extra_stages:
            next_node = "write_files"
        elif "implement_code" in extra_stages:
            next_node = "implement_code"
        else:
            next_node = None
        return StageResult(
            status=next_node if next_node else "success",
            next_node=next_node,
            message=f"Proposed {trial_id}",
            payload={"trial_id": trial_id},
            artifacts={
                "trial_dir": str(trial_dir),
                "training_config": str(training_out),
                "inference_config": str(inference_out),
                "metrics_config": str(metrics_out),
            },
        )
