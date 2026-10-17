"""F5-TTS system: the staged pipeline of an F5-TTS recipe.

On top of :class:`~espnet3.systems.base.system.BaseSystem` this adds the
two data-preparation stages an F5-TTS recipe runs between
``create_dataset`` and ``collect_stats``, ``remove_long_short`` and
``create_token_list``, and makes ``train`` leave the training config beside
the checkpoints, where inference rebuilds the model from.
"""

import logging
from pathlib import Path
from typing import ClassVar

from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.f5tts.src.create_token_list import create_token_list
from espnet3.systems.f5tts.src.remove_long_short import remove_long_short

logger = logging.getLogger(__name__)


class F5TTSSystem(BaseSystem):
    """System for recipes that train :class:`espnet3.systems.f5tts.src.f5tts.F5TTS`.

    The stage order of an F5-TTS recipe is ``create_dataset ->
    remove_long_short -> create_token_list -> collect_stats -> train ->
    infer -> measure -> pack_model -> upload_model -> pack_demo ->
    upload_demo``. The model is instantiated directly from
    ``training_config.model._target_``, with ``task`` left unset, so
    ``train`` here first writes the training config to
    ``${exp_dir}/config.yaml`` (what ``BaseSystem`` does through the ESPnet2
    task bridge for a task) and then trains; ``inference.yaml`` points
    ``model.train_config`` at that file, so a checkpoint is always rebuilt
    from the config it was trained with. Every other stage is inherited
    from ``BaseSystem`` unchanged.

    Examples:
        >>> [s.name for s in F5TTSSystem.stages[:3]]
        ['create_dataset', 'remove_long_short', 'create_token_list']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(
            name="remove_long_short",
            config="training",
            log_dir="remove_long_short.save_path",
        ),
        StageSpec(
            name="create_token_list",
            config="training",
            log_dir="create_token_list.save_path",
        ),
        StageSpec(name="collect_stats", config="training", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )

    def train(self, *args, **kwargs):
        """Write the training config beside the checkpoints, then train.

        The config is saved as ``${exp_dir}/config.yaml``, the file the
        template's ``inference.yaml`` names as ``model.train_config``, so the
        checkpoint and the config it was trained with travel together: a
        second training config needs no edit to the inference config, and a
        packed bundle carries the file with ``exp_dir``.

        Raises:
            TypeError: If any positional or keyword argument is passed.

        Example:
            .. code-block:: python

                >>> system = F5TTSSystem(training_config=training_config)
                >>> system.train()
                >>> (Path(training_config.exp_dir) / "config.yaml").is_file()
                True

        Note:
            The config is written as resolved by ``run.py``, so its paths
            are relative to the recipe directory, which is also the working
            directory ``pack_model`` copies a bundle from.
        """
        self._reject_stage_args("train", args, kwargs)
        exp_dir = Path(self.training_config.exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_path = exp_dir / "config.yaml"
        OmegaConf.save(self.training_config, config_path)
        logger.info("F5TTSSystem.train(): wrote the training config to %s", config_path)
        return super().train()

    def remove_long_short(self, *args, **kwargs):
        """Filter the recipe's manifests by audio duration.

        Runs the ``remove_long_short`` stage on ``training_config``. See
        :func:`espnet3.systems.f5tts.src.remove_long_short.remove_long_short`
        for the ``training_config.remove_long_short`` fields and the files
        written.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            RuntimeError: If required configuration is missing or a manifest
                file is not found.

        Examples:
            >>> system = F5TTSSystem(training_config=training_config)
            >>> system.remove_long_short()

        Note:
            Run it after ``create_dataset``, which writes the manifests it
            reads, and before ``create_token_list``, which reads the filtered
            training manifest.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info("F5TTSSystem.remove_long_short(): starting duration filtering")
        return remove_long_short(self.training_config)

    def create_token_list(self, *args, **kwargs):
        """Build the token list from the training manifest.

        Runs the ``create_token_list`` stage on ``training_config``. See
        :func:`espnet3.systems.f5tts.src.create_token_list.create_token_list`
        for the ``training_config.create_token_list`` fields and the file
        written.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            RuntimeError: If required configuration is missing or the
                manifest file is not found.

        Examples:
            >>> system = F5TTSSystem(training_config=training_config)
            >>> system.create_token_list()

        Note:
            The model and the dataset preprocessor both read the token list
            this stage writes, so it has to run before ``collect_stats``.
        """
        self._reject_stage_args("create_token_list", args, kwargs)
        logger.info("F5TTSSystem.create_token_list(): starting token list creation")
        return create_token_list(self.training_config)
