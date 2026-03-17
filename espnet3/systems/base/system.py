"""Base system class and stage entrypoints for ESPnet3."""

import logging
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.base.training import collect_stats, train

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

    Each system should implement the following:
      - create_dataset()
      - train()
      - infer()
      - measure()
      - publish()

    This class intentionally does NOT implement:
      - DAG
      - dependency checks
      - caching

    All behavior is config-driven.

    Args:
        training_config (DictConfig | None): Training configuration.
        inference_config (DictConfig | None): Inference configuration.
        metrics_config (DictConfig | None): Measurement configuration.
        stage_log_mapping (dict | None): Optional overrides for stage log path
            resolution. Keys are stage names; values are dotted attribute
            paths (e.g., ``"training_config.exp_dir"``) or lists/tuples of such
            paths (first non-empty value wins).

    Stage log mapping (base defaults):
        | Stage          | Path reference                     |
        |---             |---                                 |
        | create_dataset | training_config.recipe_dir         |
        | collect_stats  | training_config.stats_dir          |
        | train          | training_config.exp_dir            |
        | infer          | inference_config.inference_dir     |
        | measure        | metrics_config.inference_dir       |
        | pack_model     | training_config.exp_dir            |
        | upload_model   | training_config.exp_dir            |

    Any stage missing from the mapping (or resolving to ``None``) falls back
    to the default log directory: ``training_config.exp_dir`` when available,
    otherwise ``<cwd>/logs``.

    Examples:
        Override a subset of stage log paths:
            ```python
            system = BaseSystem(
                training_config=train_cfg,
                inference_config=infer_cfg,
                metrics_config=measure_cfg,
                stage_log_mapping={
                    "infer": "training_config.exp_dir",
                    "measure": "training_config.exp_dir",
                },
            )
            ```
    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publish_config: DictConfig | None = None,
        demo_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
    ) -> None:
        """Initialize the system with optional stage configs."""
        self.training_config = training_config
        self.inference_config = inference_config
        self.metrics_config = metrics_config
        self.publish_config = publish_config
        self.demo_config = demo_config

        if training_config is not None:
            self.exp_dir = Path(training_config.exp_dir)
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None

        if self.exp_dir is not None:
            default_dir = self.exp_dir
        else:
            default_dir = Path.cwd() / "logs"

        base_mapping = {
            "create_dataset": "training_config.recipe_dir",
            "collect_stats": "training_config.stats_dir",
            "train": "training_config.exp_dir",
            "infer": "inference_config.inference_dir",
            "measure": "metrics_config.inference_dir",
            "pack_model": "training_config.exp_dir",
            "upload_model": "training_config.exp_dir",
        }
        mapping = dict(base_mapping)
        if stage_log_mapping:
            # Explicitly override base mapping with caller-provided values.
            mapping.update(stage_log_mapping)

        self.stage_log_dirs = {"default": default_dir}
        for stage, ref in mapping.items():
            resolved = self._resolve_stage_log_ref(ref)
            if resolved:
                self.stage_log_dirs[stage] = Path(resolved)

        logger.info(
            "Initialized %s with training_config=%s inference_config=%s "
            "metrics_config=%s exp_dir=%s",
            self.__class__.__name__,
            training_config is not None,
            inference_config is not None,
            metrics_config is not None,
            self.exp_dir,
        )

    def _resolve_stage_log_ref(
        self, ref: str | list[str] | tuple[str, ...] | None
    ) -> str | None:
        """Resolve stage log mapping references to concrete values.

        Supports dotted attribute paths like ``training_config.exp_dir`` and
        fallbacks via list/tuple entries (first non-empty value wins).
        """
        target = self
        if isinstance(ref, (list, tuple)):
            for item in ref:
                resolved = self._resolve_stage_log_ref(item)
                if resolved:
                    return resolved
            return None

        if not isinstance(ref, str):
            return None

        root_name, *parts = ref.split(".")
        current = getattr(target, root_name, None)
        for part in parts:
            if current is None:
                return None
            current = getattr(current, part, None)
        return current

    @staticmethod
    def _reject_stage_args(stage: str, args, kwargs) -> None:
        """Reject unexpected positional/keyword arguments for stages."""
        if args or kwargs:
            raise TypeError(
                f"Stage '{stage}' does not accept arguments. "
                "Put all settings in the YAML config."
            )

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self, *args, **kwargs):
        """Create datasets using the configured stage."""
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("Running prepare() (BaseSystem stub). Nothing done.")

    def collect_stats(self, *args, **kwargs):
        """Collect statistics needed for training."""
        self._reject_stage_args("collect_stats", args, kwargs)
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(self.training_config, "exp_dir", None),
            getattr(self.training_config, "stats_dir", None),
        )
        return collect_stats(self.training_config)

    def train(self, *args, **kwargs):
        """Train the system model."""
        self._reject_stage_args("train", args, kwargs)
        model_target = None
        if self.training_config is not None and hasattr(self.training_config, "model"):
            model_config = self.training_config.model
            if isinstance(model_config, DictConfig):
                model_target = model_config.get("_target_")
        logger.info(
            "Training start | exp_dir=%s model=%s",
            getattr(self.training_config, "exp_dir", None),
            model_target or "<unknown>",
        )
        return train(self.training_config)

    def infer(self, *args, **kwargs):
        """Run inference on the configured datasets."""
        self._reject_stage_args("infer", args, kwargs)
        logger.info(
            "Inference start | inference_dir=%s",
            getattr(self.inference_config, "inference_dir", None),
        )
        return infer(self.inference_config)

    def measure(self, *args, **kwargs):
        """Compute evaluation metrics from hypothesis/reference outputs."""
        self._reject_stage_args("measure", args, kwargs)
        logger.info(
            "Metrics start | metrics_config=%s",
            self.metrics_config is not None,
        )
        result = measure(self.metrics_config)
        logger.info("results: %s", result)
        return result

    def publish(self, *args, **kwargs):
        """Publish artifacts from the experiment."""
        self._reject_stage_args("publish", args, kwargs)
        logger.info("Running publish(): pack_model -> upload_model")
        self.pack_model()
        return self.upload_model()

    # ---------------------------------------------------------
    # Publication stages (optional overrides)
    # ---------------------------------------------------------
    def pack_model(self, *args, **kwargs):
        """Pack model artifacts into an espnet3 bundle."""
        self._reject_stage_args("pack_model", args, kwargs)
        from espnet3.utils.publish import pack_model

        return pack_model(self)

    def upload_model(self, *args, **kwargs):
        """Upload model bundle to HuggingFace."""
        self._reject_stage_args("upload_model", args, kwargs)
        from espnet3.utils.publish import upload_model

        return upload_model(self)

    def pack_demo(self, *args, **kwargs):
        """Pack demo artifacts into a runnable demo bundle."""
        self._reject_stage_args("pack_demo", args, kwargs)
        from espnet3.demo.pack import pack_demo

        return pack_demo(self)

    def upload_demo(self, *args, **kwargs):
        """Upload demo bundle to HuggingFace Spaces (stub)."""
        self._reject_stage_args("upload_demo", args, kwargs)
        from espnet3.demo.pack import upload_demo

        return upload_demo(self)
