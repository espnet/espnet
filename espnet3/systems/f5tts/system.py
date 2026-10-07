"""F5-TTS system: the staged pipeline of an F5-TTS recipe.

On top of :class:`~espnet3.systems.base.system.BaseSystem` this adds the
two data-preparation stages an F5-TTS recipe runs between
``create_dataset`` and ``collect_stats``: ``remove_long_short`` and
``create_token_list``.
"""

import logging

from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.f5tts.create_token_list import create_token_list
from espnet3.systems.f5tts.remove_long_short import remove_long_short

logger = logging.getLogger(__name__)


class F5TTSSystem(BaseSystem):
    """System for recipes that train :class:`espnet3.systems.f5tts.f5tts.F5TTS`.

    The stage order of an F5-TTS recipe is ``create_dataset ->
    remove_long_short -> create_token_list -> collect_stats -> train ->
    infer -> measure -> pack_model -> upload_model -> pack_demo ->
    upload_demo``. Every stage other than the two added here is inherited
    from ``BaseSystem`` unchanged: the model is instantiated directly from
    ``training_config.model._target_``, with ``task`` left unset.

    Additional stage log paths:

    .. list-table::
       :header-rows: 1

       * - Stage
         - Path reference
       * - ``remove_long_short``
         - ``training_config.remove_long_short.save_path``
       * - ``create_token_list``
         - ``training_config.create_token_list.save_path``
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
        """Initialize the F5-TTS system with optional stage configs.

        Args:
            training_config: Training configuration. Also carries the
                ``remove_long_short`` and ``create_token_list`` blocks.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for model packing
                and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides,
                merged over the two entries this system adds.
            demo_config: Demo configuration for demo packing and upload
                stages.

        Example:
            .. code-block:: python

                >>> system = F5TTSSystem(training_config=training_config)
                >>> str(system.stage_log_dirs["remove_long_short"])
                'data/manifest_filtered'
                >>> str(system.stage_log_dirs["create_token_list"])
                'data/token_list'

        Note:
            An entry in ``stage_log_mapping`` wins over the default for the
            same stage, so a subclass can redirect a stage's log or add
            entries for stages of its own.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "remove_long_short": "training_config.remove_long_short.save_path",
                "create_token_list": "training_config.create_token_list.save_path",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    def remove_long_short(self, *args, **kwargs):
        r"""Filter the recipe's manifests by audio duration.

        Runs the ``remove_long_short`` stage on ``training_config``. See
        :func:`espnet3.systems.f5tts.remove_long_short.remove_long_short`
        for the ``training_config.remove_long_short`` fields and the files
        written.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            RuntimeError: If required configuration is missing or a manifest
                file is not found.

        Example:
            .. code-block:: python

                >>> system = F5TTSSystem(training_config=training_config)
                >>> system.remove_long_short()

            From the command line, through a recipe's ``run.py``:

            .. code-block:: bash

                python run.py --stages remove_long_short \
                    --training_config conf/training.yaml

        Note:
            Run it after ``create_dataset``, which writes the manifests it
            reads, and before ``create_token_list``, which reads the filtered
            training manifest.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info("F5TTSSystem.remove_long_short(): starting duration filtering")
        return remove_long_short(self.training_config)

    def create_token_list(self, *args, **kwargs):
        r"""Build the token list from the training manifest.

        Runs the ``create_token_list`` stage on ``training_config``. See
        :func:`espnet3.systems.f5tts.create_token_list.create_token_list`
        for the ``training_config.create_token_list`` fields and the file
        written.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            RuntimeError: If required configuration is missing or the
                manifest file is not found.

        Example:
            .. code-block:: python

                >>> system = F5TTSSystem(training_config=training_config)
                >>> system.create_token_list()

            From the command line, through a recipe's ``run.py``:

            .. code-block:: bash

                python run.py --stages create_token_list \
                    --training_config conf/training.yaml

        Note:
            The model and the dataset preprocessor both read the token list
            this stage writes, so it has to run before ``collect_stats``.
        """
        self._reject_stage_args("create_token_list", args, kwargs)
        logger.info("F5TTSSystem.create_token_list(): starting token list creation")
        return create_token_list(self.training_config)
