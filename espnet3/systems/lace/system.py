"""LACE system: the staged pipeline of a LACE recipe.

On top of :class:`~espnet3.systems.base.system.BaseSystem` this adds what a
LACE (Layer-Adaptive Codec Encoding) run needs in ``collect_stats`` and
``train``:

- The model is an espnet2 GAN codec with a wrapped quantizer
  (``AbsGANESPnetModel``), so both stages build the generator/discriminator
  trainer from :mod:`espnet3.systems.lace.gan_trainer` instead of the plain
  single-optimizer trainer.
- The model is built through a Hydra ``_target_``
  (``espnet3.systems.lace.lace.LACE``) with ``task:`` unset, so
  there is no espnet2-style ``config.yaml`` for inference to rebuild from.
  ``train`` writes the equivalent, ``${exp_dir}/lace_model.yaml``: the
  ``model`` block as ``LACE`` keyword arguments, which
  ``LACEAudioCoding`` passes back to ``LACE`` before loading
  the fine-tuned checkpoint.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Union

import lightning as L
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.lace.gan_trainer import build_gan_trainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_NAME = "lace_model.yaml"


def build_lace_model_config(model_config: Union[DictConfig, Dict[str, Any]]):
    """Turn a training config's ``model`` block into ``LACE`` constructor kwargs.

    Args:
        model_config: The ``model`` block of a LACE training config, i.e. a
            Hydra node whose ``_target_`` is ``LACE``.

    Returns:
        Plain dict of ``LACE`` keyword arguments: ``_target_``
        and the other Hydra instantiate keys (``_convert_``, ``_partial_``,
        ...) are removed, and interpolations are resolved.

    Raises:
        ValueError: If ``model_config`` has no ``_target_``; a task-built
            model block (``codec``/``codec_conf`` under ``task:``) is not a
            LACE model and has no spec to write.

    Examples:
        ```python
        from omegaconf import OmegaConf

        model_config = OmegaConf.create({
            "_target_": "espnet3.systems.lace.lace.LACE",
            "_convert_": "all",
            "pretrained_model_tag": "NewGame/libritts_codec_train_encodec_libritts",
            "compression_model": {"name": "cosine_similarity"},
        })
        build_lace_model_config(model_config)
        # -> {'pretrained_model_tag': 'NewGame/libritts_codec_train_encodec_libritts',
        #     'compression_model': {'name': 'cosine_similarity'}}
        ```
    """
    if OmegaConf.is_config(model_config):
        spec = OmegaConf.to_container(model_config, resolve=True)
    else:
        spec = dict(model_config)
    if spec.get("_target_") is None:
        raise ValueError(
            "LACE training configs build the model through `model._target_` "
            "(espnet3.systems.lace.lace.LACE) with `task:` unset; "
            f"got model keys {sorted(spec)}."
        )
    # Hydra's own keys (`_target_`, plus `_convert_: all`, which the recipe
    # config loader adds to every target node) are not builder arguments.
    return {key: value for key, value in spec.items() if not _is_hydra_key(key)}


def _is_hydra_key(key: str) -> bool:
    return key.startswith("_") and key.endswith("_")


def write_lace_model_config(
    model_config: Union[DictConfig, Dict[str, Any]], output_path: Union[str, Path]
) -> Path:
    """Write the ``LACE`` constructor spec of a training config to YAML.

    Args:
        model_config: The ``model`` block of a LACE training config.
        output_path: Destination file; parent directories are created.

    Returns:
        The written path.

    Raises:
        ValueError: Propagated from :func:`build_lace_model_config`.

    Examples:
        ```python
        write_lace_model_config(config.model, "exp/train_encodec_lace/lace_model.yaml")
        ```
    """
    spec = build_lace_model_config(model_config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return output_path


class LACESystem(BaseSystem):
    """System for recipes that fine-tune a codec with the LACE wrapper.

    The stage order of a LACE recipe is the default one: ``create_dataset ->
    collect_stats -> train -> infer -> measure -> pack_model -> upload_model
    -> pack_demo -> upload_demo``. ``create_dataset``, ``infer``, ``measure``
    and the publication stages are inherited from ``BaseSystem`` unchanged.
    ``collect_stats`` and ``train`` are overridden only to build the GAN
    trainer, and ``train`` additionally writes ``${exp_dir}/lace_model.yaml``
    before fitting. The model is instantiated directly from
    ``training_config.model._target_``; ``task`` must be left unset.

    Examples:
        Wired through ``egs3/TEMPLATE/lace/run.py``:
        ```python
        from egs3.TEMPLATE.lace.run import DEFAULT_STAGES, build_parser, main
        from espnet3.systems.lace.system import LACESystem

        parser = build_parser(stages=DEFAULT_STAGES)
        args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
        main(args=args, system_cls=LACESystem, stages=DEFAULT_STAGES)
        ```
    """

    def _reject_task_config(self) -> None:
        task = self.training_config.get("task")
        if task:
            raise ValueError(
                "LACE training configs must leave `task:` unset so the model is "
                f"built from `model._target_`; got task={task!r}."
            )

    def _build_trainer(self) -> ESPnet3LightningTrainer:
        config = self.training_config
        model = instantiate(config.model)
        if isinstance(model, AbsGANESPnetModel):
            return build_gan_trainer(config, model)

        lit_model = ESPnetLightningModule(model, config)
        return ESPnet3LightningTrainer(
            model=lit_model,
            exp_dir=config.exp_dir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

    def _prepare_training_runtime(self) -> None:
        config = self.training_config
        Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
        # `stats_dir` may be declared but left null (the recipe skips
        # collect_stats), so check the value rather than key presence.
        stats_dir = config.get("stats_dir")
        if stats_dir is not None:
            Path(stats_dir).mkdir(parents=True, exist_ok=True)

        if config.get("parallel"):
            set_parallel(config.parallel)

        if config.get("seed") is not None:
            L.seed_everything(int(config.seed), workers=True)

        torch.set_float32_matmul_precision("high")

    def collect_stats(self, *args, **kwargs):
        """Run the collect_stats stage with the GAN trainer.

        Args:
            *args: Not accepted. Stage settings belong in the YAML config.
            **kwargs: Not accepted, for the same reason.

        Returns:
            None. Statistics are written under ``training_config.stats_dir``.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            ValueError: If ``training_config.task`` is set.

        Examples:
            Invoked by ``run.py`` for the ``collect_stats`` stage:
            ```python
            system = LACESystem(training_config=config)
            system.collect_stats()
            ```
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        self._reject_task_config()
        start = time.perf_counter()
        self._prepare_training_runtime()

        trainer = self._build_trainer()
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            self.training_config.get("stats_dir"),
        )

    def train(self, *args, **kwargs):
        """Write ``lace_model.yaml`` into ``exp_dir``, then fit with the GAN trainer.

        Args:
            *args: Not accepted. Stage settings belong in the YAML config.
            **kwargs: Not accepted, for the same reason.

        Returns:
            None. ``${exp_dir}/lace_model.yaml``, checkpoints and logs are
            written under ``training_config.exp_dir``.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            ValueError: If ``training_config.task`` is set or
                ``training_config.model`` has no ``_target_``, i.e. the
                config does not build a LACE model.

        Examples:
            Invoked by ``run.py`` for the ``train`` stage:
            ```python
            system = LACESystem(training_config=config)
            system.train()
            ```

            Extra keyword arguments for ``trainer.fit`` come from the
            config's ``fit`` block, not from this call:
            ```yaml
            fit:
              ckpt_path: exp/train_encodec_lace_libritts/last.ckpt
            ```
        """
        self._reject_stage_args("train", args, kwargs)
        self._reject_task_config()
        start = time.perf_counter()
        config = self.training_config
        self._prepare_training_runtime()

        output_path = write_lace_model_config(
            config.model, Path(config.exp_dir) / MODEL_CONFIG_NAME
        )
        logger.info("Wrote LACE model spec to %s", output_path)

        trainer = self._build_trainer()

        fit_kwargs: Dict[str, Any] = {}
        if config.get("fit"):
            fit_kwargs = OmegaConf.to_container(config.fit, resolve=True)

        trainer.fit(**fit_kwargs)
        logger.info(
            "Training finished in %.2fs | exp_dir=%s",
            time.perf_counter() - start,
            config.exp_dir,
        )
