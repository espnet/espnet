"""Voice conversion system.

Adds the ``prepare_features`` stage on top of :class:`BaseSystem`: it
precomputes frozen self-supervised features (WavLM for kNN-VC), optionally
prematched within each pool the recipe declares through
``Dataset.get_pool_key`` (a speaker chapter for kNN-VC), that the vocoder
``train`` stage consumes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar

from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.knnvc.src.prepare_features_provider import (
    PrepareFeaturesProvider,
)
from espnet3.systems.knnvc.src.prepare_features_runner import PrepareFeaturesRunner

logger = logging.getLogger(__name__)


class KNNVCSystem(BaseSystem):
    """Voice-conversion system (encoder -> converter -> vocoder recipes).

    The declared stage order, which ``launch(KNNVCSystem)`` runs for
    ``--stages all``::

        create_dataset -> prepare_features -> train -> infer -> measure
        -> pack_model -> upload_model

    Two stages of :class:`BaseSystem` are left out on purpose:

    - ``collect_stats``: the HiFi-GAN vocoder needs no feature statistics;
    - ``pack_demo`` / ``upload_demo``: a VC demo takes a variable-length set
      of target-speaker reference utterances as one input, which the shared
      demo UI asset types (``audio``, ``text``) cannot express yet.

    ``prepare_features`` logs under
    ``training_config.prepare_features.features_dir``.

    Examples:
        >>> [s.name for s in KNNVCSystem.stages[:3]]
        ['create_dataset', 'prepare_features', 'train']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(
            name="prepare_features",
            config="training",
            log_dir="prepare_features.features_dir",
        ),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
    )

    def prepare_features(self, *args, **kwargs):
        """Precompute (prematched) encoder features for vocoder training.

        Port of kNN-VC's ``prematch_dataset.py``. Every utterance of every
        listed dataset is encoded with the frozen encoder; with ``prematch``
        each frame is additionally replaced by the mean of its ``topk``
        nearest frames among the *other* utterances sharing its pool key
        (``dataset.get_pool_key(idx)``; kNN-VC's official script pools per
        speaker chapter directory, and the recipe can pool per speaker
        instead).
        Work is dispatched through ``PrepareFeaturesRunner`` /
        ``PrepareFeaturesProvider`` and therefore honours
        ``training_config.parallel``.

        Under ``training_config.prepare_features``, configure:

        .. list-table::
           :header-rows: 1

           * - Field
             - Description
           * - ``features_dir``
             - Output directory. Files are written to
               ``<features_dir>/<feature_name>.npy`` (float16) plus one merged
               ``<features_dir>/feats.<name>.scp`` per dataset entry; shard
               bookkeeping lives in ``<features_dir>/shards/<name>/``.
           * - ``dataset``
             - List of dataset reference entries (``name``, ``data_src``,
               ``data_src_args``) yielding items with a ``speech`` waveform.
               The dataset class must implement ``get_pool_key(idx)`` and
               ``get_feature_name(idx)`` (see ``PrepareFeaturesProvider``).
           * - ``encoder``
             - Hydra config of the encoder, e.g.
               ``espnet3.systems.knnvc.src.wavlm_encoder.WavLMEncoder``.
           * - ``prematch``
             - Replace each frame by the mean of its ``topk`` nearest frames
               within the same pool (default ``true``).
           * - ``topk``
             - ``k`` for prematching (default ``4``).
           * - ``device``
             - Encoder device; defaults to CUDA when available.
           * - ``batch_size``
             - Optional number of indices per ``forward`` call.

        Example:
            .. code-block:: yaml

                prepare_features:
                  features_dir: ${data_dir}/wavlm_l6_prematched
                  dataset:
                    - name: train-clean-100
                      data_src_args: {split: train-clean-100, kind: audio}
                  encoder:
                    _target_: espnet3.systems.knnvc.src.wavlm_encoder.WavLMEncoder
                    checkpoint: /path/to/WavLM-Large.pt
                    layer: 6
                  prematch: true
                  topk: 4

        Re-running is safe: completed shards are skipped and feature files are
        replaced atomically.

        Raises:
            RuntimeError: If required configuration is missing.
            TypeError: If a dataset does not implement the stage contract.
        """
        self._reject_stage_args("prepare_features", args, kwargs)
        logger.info("KNNVCSystem.prepare_features(): starting feature preparation")
        start = time.perf_counter()

        stage_config = self._get_required_config(
            self.training_config,
            "prepare_features",
            "training_config.prepare_features must be set for prepare_features stage.",
        )
        features_dir = Path(
            self._get_required_config(
                stage_config,
                "features_dir",
                "training_config.prepare_features.features_dir must be set.",
            )
        )
        dataset_entries = self._get_required_config(
            stage_config,
            "dataset",
            "training_config.prepare_features.dataset must list at least one "
            "dataset reference entry.",
        )
        encoder_config = self._get_required_config(
            stage_config,
            "encoder",
            "training_config.prepare_features.encoder must be a Hydra config.",
        )
        if OmegaConf.is_config(dataset_entries):
            dataset_entries = OmegaConf.to_container(dataset_entries, resolve=True)
        if isinstance(dataset_entries, dict):
            dataset_entries = [dataset_entries]
        if not dataset_entries:
            raise RuntimeError(
                "training_config.prepare_features.dataset must not be empty."
            )

        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        features_dir.mkdir(parents=True, exist_ok=True)
        prematch = bool(stage_config.get("prematch", True))
        topk = int(stage_config.get("topk", 4))
        logger.info(
            "prepare_features | features_dir=%s prematch=%s topk=%d datasets=%d",
            features_dir,
            prematch,
            topk,
            len(dataset_entries),
        )

        for index, entry in enumerate(dataset_entries):
            entry = dict(entry)
            name = str(entry.pop("name", None) or f"dataset{index}")
            provider = PrepareFeaturesProvider(
                config=self.training_config,
                params={
                    "dataset": entry,
                    "encoder": encoder_config,
                    "features_dir": str(features_dir),
                    "prematch": prematch,
                    "topk": topk,
                    "device": stage_config.get("device", None),
                },
            )
            # Sort by pool key so each worker sees one pool's utterances in a
            # row and can reuse its encoded pool (see PrepareFeaturesRunner).
            pool_indices = PrepareFeaturesProvider.build_pool_indices(provider.dataset)
            indices = [idx for key in sorted(pool_indices) for idx in pool_indices[key]]
            logger.info(
                "Dataset '%s': %d utterances in %d prematching pools",
                name,
                len(indices),
                len(pool_indices),
            )
            runner = PrepareFeaturesRunner(
                provider=provider,
                batch_size=stage_config.get("batch_size", None),
                output_dir=features_dir / "shards",
                shard_subdir=name,
            )
            if not indices:
                raise RuntimeError(
                    f"prepare_features dataset entry '{name}' resolved to an "
                    "empty dataset, so no features would be written. Check its "
                    "`data_src_args` (split/kind/subset)."
                )
            scp_path = runner(indices)
            logger.info("Dataset '%s' done -> %s", name, scp_path)

        logger.info(
            "prepare_features finished in %.2fs | features_dir=%s",
            time.perf_counter() - start,
            features_dir,
        )
