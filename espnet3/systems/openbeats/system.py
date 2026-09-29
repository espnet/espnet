"""OpenBEATs iterative self-supervised pre-training system."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import train as train_model

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "<unk>"


class OpenBeatsSystem(BaseSystem):
    r"""System for BEATs iterative pre-training (arXiv:2212.09058).

    One ``pretrain`` run trains one BEATs *iteration*, selected by
    ``training_config.iteration``:

    - iteration 0: tokenize the corpus with a random-projection tokenizer and
      train the encoder against those targets.
    - iteration N > 0: train a VQ tokenizer distilled from the iteration N-1
      encoder, re-tokenize the corpus with it, and train the encoder again.

    Stages (canonical order, see ``egs3/TEMPLATE/openbeats/run.py``):

    | Stage             | Config                    | What it does                    |
    |---                |---                        |---                              |
    | `pretrain`        | all of the below          | Runs the next five stages       |
    | `create_dataset`  | `training_config`         | Recipe `DatasetBuilder`         |
    | `train_tokenizer` | `train_tokenizer_config`  | No-op at iteration 0; otherwise |
    |                   |                           | trains the tokenizer            |
    | `infer`           | `inference_config`        | Writes `target.scp` and         |
    |                   |                           | `target_shape` per test set     |
    | `collect_stats`   | `training_config`         | Writes `feats_shape` files      |
    | `train`           | `training_config`         | Trains the encoder              |
    | `measure`         | `metrics_config`          | Scores the targets, e.g. with   |
    |                   |                           | `CodebookUsage`                 |
    | `pack_model`,     | `publication_config`      | Packs / uploads the exported    |
    | `upload_model`    |                           | encoder                         |

    The portable checkpoints (``beats_encoder_iter<N>.pt`` and
    ``beats_tokenizer_iter<N>.pt``) are written at the end of training by
    :class:`~espnet3.systems.openbeats.callbacks.BeatsCheckpointExport`,
    configured under ``trainer.callbacks`` of the two training configs.

    Config fields read by this class, on top of the ``BaseSystem`` ones:

    - ``training_config.iteration`` (int, default ``0``).
    - ``training_config.model.token_list`` and
      ``training_config.model.encoder_conf.beats_config.codebook_vocab_size``.
    - ``training_config.target_dir``: must equal
      ``inference_config.inference_dir`` so training reads the targets that
      ``infer`` wrote.
    - ``train_tokenizer_config.export_path`` and
      ``train_tokenizer_config.model.beats_teacher_ckpt_path`` (iteration > 0).
    - ``inference_config.model.tokenizer_ckpt_path``: filled with
      ``train_tokenizer_config.export_path`` when left ``null`` at
      iteration > 0.

    Additional stage log paths:
        | Stage           | Path reference                  |
        |---              |---                              |
        | train_tokenizer | train_tokenizer_config.exp_dir  |

    Args:
        training_config: Encoder training configuration.
        inference_config: Tokenization configuration for the ``infer`` stage.
        metrics_config: Measurement configuration for the ``measure`` stage.
        publication_config: Publication configuration.
        stage_log_mapping: Optional per-stage log directory overrides.
        demo_config: Unused; OpenBEATs has no demo stages.
        train_tokenizer_config: Tokenizer training configuration, required for
            ``train_tokenizer`` at iteration > 0.

    Examples:
        Iteration 0, then iteration 1 (see ``egs3/mini_an4/openbeats/readme.md``):

        .. code-block:: bash

            python run.py --stages pretrain measure \
                --training_config conf/training.yaml \
                --inference_config conf/inference.yaml \
                --metrics_config conf/metrics.yaml
            python run.py --stages pretrain measure \
                --training_config conf/training_iter1.yaml \
                --train_tokenizer_config conf/training_tokenizer.yaml \
                --inference_config conf/inference.yaml \
                --metrics_config conf/metrics.yaml
    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
        demo_config: DictConfig | None = None,
        train_tokenizer_config: DictConfig | None = None,
    ) -> None:
        """Initialize the OpenBEATs system with the per-stage configs."""
        # Set before BaseSystem.__init__ so the stage log mapping can resolve
        # `train_tokenizer_config.exp_dir`.
        self.train_tokenizer_config = train_tokenizer_config
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "train_tokenizer": "train_tokenizer_config.exp_dir",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    # ---------------------------------------------------------
    # Stages
    # ---------------------------------------------------------
    def pretrain(self, *args, **kwargs):
        """Run one BEATs iteration end to end.

        Runs ``create_dataset``, ``train_tokenizer``, ``infer``,
        ``collect_stats``, and ``train`` in that order. ``collect_stats`` is
        skipped when the shape statistics already exist, because they do not
        depend on the iteration.

        ``pretrain`` needs a single device. Lightning's DDP launcher re-runs
        the invoked stages in every rank, so with ``num_device > 1`` the
        non-training steps (tokenization, statistics) would run once per rank
        and the second ``fit`` would start a second set of ranks. Run the five
        stages as separate invocations instead (see the recipe readme).

        Raises:
            RuntimeError: If ``num_device * num_nodes > 1``.
        """
        self._reject_stage_args("pretrain", args, kwargs)
        num_ranks = int(self.training_config.get("num_device", 1) or 1) * int(
            self.training_config.get("num_nodes", 1) or 1
        )
        if num_ranks > 1:
            raise RuntimeError(
                f"pretrain runs on a single device, but num_device * num_nodes = "
                f"{num_ranks}. Run `create_dataset`, `train_tokenizer`, `infer`, "
                "`collect_stats`, and `train` as separate run.py invocations; "
                "see the recipe readme."
            )
        self.create_dataset()
        self.train_tokenizer()
        self.infer()
        if self._has_shape_stats():
            logger.info(
                "Shape statistics already exist in %s; skipping collect_stats().",
                self.training_config.stats_dir,
            )
        else:
            self.collect_stats()
        return self.train()

    def train_tokenizer(self, *args, **kwargs):
        """Train the BEATs VQ tokenizer for iterations greater than 0.

        The tokenizer is distilled from the teacher encoder at
        ``train_tokenizer_config.model.beats_teacher_ckpt_path`` (normally
        ``beats_encoder_iter<N-1>.pt``). The ``BeatsCheckpointExport``
        callback writes ``train_tokenizer_config.export_path`` when training
        ends. Re-running is safe: the stage is skipped when that file exists.

        Raises:
            RuntimeError: If ``train_tokenizer_config`` is missing.
            FileNotFoundError: If the teacher checkpoint does not exist.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)
        iteration = self._get_iteration()
        if iteration == 0:
            logger.info(
                "Iteration 0 uses the random-projection tokenizer; "
                "skipping train_tokenizer()."
            )
            return None

        export_path = self._get_tokenizer_checkpoint_path()
        if export_path.exists():
            logger.info("Tokenizer already exported: %s. Skipping.", export_path)
            return None

        config = self.train_tokenizer_config
        teacher = self._get_required_config(
            config.model,
            "beats_teacher_ckpt_path",
            "train_tokenizer_config.model.beats_teacher_ckpt_path must be set.",
        )
        if not Path(teacher).is_file():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher}. Run the `train` stage "
                f"for iteration {iteration - 1} first."
            )
        logger.info(
            "Training BEATs tokenizer | iteration=%d teacher=%s exp_dir=%s",
            iteration,
            teacher,
            config.exp_dir,
        )
        # BaseSystem.train() always trains training_config; the tokenizer has
        # its own config, so call the shared training entrypoint directly.
        return train_model(config)

    def infer(self, *args, **kwargs):
        """Tokenize the configured test sets into BEATs training targets.

        Writes two files per ``inference_config.dataset.test`` entry:

        - ``<inference_dir>/<test_name>/target.scp``: dataset index followed
          by the token ids of that item.
        - ``<inference_dir>/<test_name>/target_shape``: dataset index and
          number of token ids, used as a batching shape file.

        At iteration > 0, ``inference_config.model.tokenizer_ckpt_path``
        defaults to this iteration's exported tokenizer.

        Examples:
            First lines of ``targets/train/target.scp`` and ``target_shape``
            for 10 s AudioSet clips (496 patches each)::

                0 886 468 468 468 280 442 ...
                1 468 874 280 280 418 280 ...

                0 496
                1 496

        Raises:
            ValueError: If ``training_config.target_dir`` and
                ``inference_config.inference_dir`` differ.
            FileNotFoundError: If the tokenizer checkpoint is missing.
        """
        self._reject_stage_args("infer", args, kwargs)
        config = self.inference_config
        self._validate_target_dir()
        iteration = self._get_iteration()
        model_config = config.model
        if iteration > 0 and not model_config.get("tokenizer_ckpt_path"):
            model_config.tokenizer_ckpt_path = str(
                self._get_tokenizer_checkpoint_path()
            )
        tokenizer_ckpt_path = model_config.get("tokenizer_ckpt_path")
        if tokenizer_ckpt_path and not Path(tokenizer_ckpt_path).is_file():
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found: {tokenizer_ckpt_path}. Run the "
                f"`train_tokenizer` stage for iteration {iteration} first."
            )
        logger.info(
            "Tokenizing with %s tokenizer | iteration=%d",
            tokenizer_ckpt_path or "random-projection",
            iteration,
        )
        result = super().infer()
        for test_set in config.dataset.test:
            test_dir = Path(config.inference_dir) / test_set.name
            _write_target_shape(test_dir / "target.scp", test_dir / "target_shape")
        return result

    def collect_stats(self, *args, **kwargs):
        """Write the token list if needed, then collect shape statistics."""
        self._reject_stage_args("collect_stats", args, kwargs)
        self._ensure_token_list()
        return super().collect_stats()

    def train(self, *args, **kwargs):
        """Train the BEATs encoder of this iteration.

        The ``BeatsCheckpointExport`` callback writes
        ``training_config.export_path`` (``beats_encoder_iter<N>.pt``) when
        training ends.
        """
        self._reject_stage_args("train", args, kwargs)
        self._ensure_token_list()
        return super().train()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _get_iteration(self) -> int:
        """Return the BEATs iteration selected by ``training_config.iteration``."""
        if self.training_config is None:
            return 0
        iteration = int(self.training_config.get("iteration", 0) or 0)
        if iteration < 0:
            raise ValueError(f"training_config.iteration must be >= 0: {iteration}")
        return iteration

    def _get_tokenizer_checkpoint_path(self) -> Path:
        """Return the tokenizer checkpoint exported for this iteration."""
        config = self._get_required_config(
            {"train_tokenizer_config": self.train_tokenizer_config},
            "train_tokenizer_config",
            "--train_tokenizer_config is required at iteration > 0.",
        )
        return Path(
            self._get_required_config(
                config,
                "export_path",
                "train_tokenizer_config.export_path must be set.",
            )
        )

    def _has_shape_stats(self) -> bool:
        stats_dir = Path(self.training_config.stats_dir)
        return all(
            (stats_dir / mode / "feats_shape").is_file() for mode in ("train", "valid")
        )

    def _validate_target_dir(self) -> None:
        if self.training_config is None:
            return
        target_dir = self.training_config.get("target_dir")
        inference_dir = self.inference_config.get("inference_dir")
        if target_dir is None or inference_dir is None:
            return
        if Path(target_dir).resolve() != Path(inference_dir).resolve():
            raise ValueError(
                "training_config.target_dir must match inference_config."
                f"inference_dir so training reads the written targets: "
                f"{target_dir} != {inference_dir}"
            )

    def _ensure_token_list(self) -> None:
        model_config = self.training_config.model
        token_list = Path(
            self._get_required_config(
                model_config,
                "token_list",
                "training_config.model.token_list must be set.",
            )
        )
        codebook_size = int(
            model_config.encoder_conf.beats_config.get("codebook_vocab_size", 1024)
        )
        _write_token_list(token_list, codebook_size)


def _write_token_list(output_path: str | Path, codebook_size: int) -> Path:
    """Write the BEATs token list: ``<unk>`` followed by the codebook ids.

    The ``<unk>`` entry keeps ids 1-based after ``CommonPreprocessor`` word
    tokenization; ``BeatsPretrainModel`` subtracts 1 before the loss. An
    existing file is kept when it has the expected content.

    For ``codebook_size=1024`` the file starts with::

        <unk>
        0
        1

    and ends with ``1023`` (1025 lines).

    Raises:
        ValueError: If an existing file does not match ``codebook_size``.
    """
    output = Path(output_path)
    expected = [UNKNOWN_TOKEN] + [str(i) for i in range(codebook_size)]
    if output.exists():
        existing = output.read_text(encoding="utf-8").splitlines()
        if existing != expected:
            raise ValueError(
                f"Existing token list {output} does not match codebook size "
                f"{codebook_size}. Remove it or fix codebook_vocab_size."
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    # Every rank runs collect_stats/train, so the temporary name must be
    # per-process: a shared one is truncated by the next writer while an
    # earlier rank is still writing it.
    tmp_path = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    tmp_path.write_text("\n".join(expected) + "\n", encoding="utf-8")
    tmp_path.replace(output)
    logger.info("Wrote token list (%d entries): %s", len(expected), output)
    return output


def _write_target_shape(target_path: str | Path, output_path: str | Path) -> Path:
    """Write a batching shape file (``<idx> <num_tokens>``) from ``target.scp``.

    ESPnet's length batch sampler reads it together with ``collect_stats``'
    ``feats_shape``. For a ``target.scp`` starting with::

        0 886 468 468
        1 874 280

    the shape file starts with::

        0 3
        1 2
    """
    output = Path(output_path)
    # Per-process temporary name, for the same reason as in _write_token_list.
    tmp_path = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    with (
        Path(target_path).open("r", encoding="utf-8") as reader,
        tmp_path.open("w", encoding="utf-8") as writer,
    ):
        for line in reader:
            key, _, tokens = line.rstrip("\n").partition(" ")
            writer.write(f"{key} {len(tokens.split())}\n")
    tmp_path.replace(output)
    return output
