import logging

import lightning as L

from omegaconf import (
    DictConfig,
    OmegaConf
)
from pathlib import Path

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.asr.system import ASRSystem


from egs3.librispeech_100.asr.src.collect_stats import collect_stats

logger = logging.getLogger(__name__)

class LBSSystem(ASRSystem):
    """ASR-specific system.

    This system adds:
      - Collect locally statistics

    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
        demo_config: DictConfig | None = None,
    ) -> None:
        """Initialize the ASR system with optional stage configs.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for model packing
                and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides.
            demo_config: Demo configuration for demo packing and upload
                stages.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "train_tokenizer": "training_config.tokenizer.save_path",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    def collect_stats(self, *args, **kwargs):
        """Collect statistics needed for training."""
        self._reject_stage_args("collect_stats", args, kwargs)
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(self.training_config, "exp_dir", None),
            getattr(self.training_config, "stats_dir", None),
        )

        Path(self.training_config.exp_dir).mkdir(parents=True, exist_ok=True)

        assert hasattr(self.training_config, "stats_dir"), "training_config.stats_dir must be defined"
        Path(self.training_config.stats_dir).mkdir(parents=True, exist_ok=True)

        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        if self.training_config.get("seed") is not None:
            L.seed_everything(int(self.training_config.seed), workers=True)

        if "normalize" in self.training_config.model:
            self.training_config.model.pop("normalize")
        if "normalize_conf" in self.training_config.model:
            self.training_config.model.pop("normalize_conf")


        # Detach dataset/dataloader configs from the root so interpolations like
        # ${dataset_dir} remain resolved when used standalone during collection.
        dataset_config = OmegaConf.create(
            OmegaConf.to_container(self.training_config.dataset, resolve=True)
        )

        dataloader_config = OmegaConf.create(
            OmegaConf.to_container(self.training_config.dataloader, resolve=True)
        )

        for mode in ["train", "valid"]:
            if mode == "train":
                dataset_config.preprocessor.train = True
            else:
                dataset_config.preprocessor.train = False

            collect_stats(
                model_config=OmegaConf.to_container(self.training_config.model, resolve=True),
                dataset_config=dataset_config,
                dataloader_config=dataloader_config,
                mode=mode,
                output_dir=Path(self.training_config.stats_dir),
                task=getattr(self.training_config, "task", None),
                parallel_config=(
                    None
                    if "parallel" not in self.training_config.keys()
                    else self.training_config.parallel
                ),
                write_collected_feats=True,
            )