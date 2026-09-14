"""BEATs iterative self-supervised pre-training system."""

from __future__ import annotations

import logging
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import train as run_training
from espnet3.systems.ssl.checkpoint_export import export_beats_checkpoint

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "<unk>"


class BeatsSystem(BaseSystem):
    """System for BEATs iterative pre-training (arXiv:2212.09058).

    One ``run.py`` invocation runs one BEATs *iteration*, selected by
    ``training_config.iteration``:

    - iteration 0: tokenize the corpus with a random-projection tokenizer and
      train the encoder against those targets.
    - iteration N > 0: train a VQ tokenizer distilled from the iteration N-1
      encoder, re-tokenize the corpus with it, and train the encoder again.

    Stages (canonical order, see ``egs3/TEMPLATE/ssl/run.py``):

    | Stage             | Config                    | What it does                    |
    |---                |---                        |---                              |
    | `create_dataset`  | `training_config`         | Recipe `DatasetBuilder`         |
    | `train_tokenizer` | `train_tokenizer_config`  | No-op at iteration 0; otherwise |
    |                   |                           | trains the tokenizer and writes |
    |                   |                           | `beats_tokenizer_iter<N>.pt`    |
    | `infer`           | `inference_config`        | Writes `target.scp` and         |
    |                   |                           | `target_shape` per test set     |
    | `collect_stats`   | `training_config`         | Writes `feats_shape` files      |
    | `train`           | `training_config`         | Trains the encoder and writes   |
    |                   |                           | `beats_encoder_iter<N>.pt`      |

    ``collect_stats`` and ``train`` also write the token list
    (``training_config.model.token_list``: ``<unk>`` followed by the codebook
    ids) when it does not exist yet.

    Config fields read by this class, on top of the ``BaseSystem`` ones:

    - ``training_config.iteration`` (int, default ``0``).
    - ``training_config.model.token_list`` and
      ``training_config.model.encoder_conf.beats_config.codebook_vocab_size``.
    - ``training_config.target_dir``: must equal
      ``inference_config.inference_dir`` so training reads the targets that
      ``infer`` wrote.
    - ``train_tokenizer_config.exp_dir`` and
      ``train_tokenizer_config.model.beats_teacher_ckpt_path`` (iteration > 0).
    - ``inference_config.model.tokenizer_ckpt_path``: filled with the
      iteration's tokenizer checkpoint when left ``null`` at iteration > 0.

    Additional stage log paths:
        | Stage           | Path reference                  |
        |---              |---                              |
        | train_tokenizer | train_tokenizer_config.exp_dir  |

    Args:
        training_config: Encoder training configuration.
        inference_config: Tokenization configuration for the ``infer`` stage.
        metrics_config: Unused by BEATs pre-training; accepted for the
            ``run.py`` protocol.
        publication_config: Publication configuration.
        stage_log_mapping: Optional per-stage log directory overrides.
        demo_config: Demo configuration.
        train_tokenizer_config: Tokenizer training configuration, required for
            ``train_tokenizer`` at iteration > 0.

    Examples:
        Iteration 0, then iteration 1 (see ``egs3/mini_an4/ssl/readme.md``):

        .. code-block:: bash

            python run.py --stages create_dataset infer collect_stats train
                --training_config conf/training.yaml
                --inference_config conf/inference.yaml
            python run.py --stages train_tokenizer infer train
                --training_config conf/training_iter1.yaml
                --train_tokenizer_config conf/training_tokenizer.yaml
                --inference_config conf/inference.yaml
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
        """Initialize the BEATs system with the per-stage configs."""
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

    @property
    def iteration(self) -> int:
        """Return the BEATs iteration selected by ``training_config.iteration``."""
        if self.training_config is None:
            return 0
        iteration = int(self.training_config.get("iteration", 0) or 0)
        if iteration < 0:
            raise ValueError(f"training_config.iteration must be >= 0: {iteration}")
        return iteration

    def get_encoder_checkpoint_path(self) -> Path:
        """Return the portable encoder checkpoint path of this iteration."""
        return Path(self.training_config.exp_dir) / (
            f"beats_encoder_iter{self.iteration}.pt"
        )

    def get_tokenizer_checkpoint_path(self) -> Path:
        """Return the portable tokenizer checkpoint path of this iteration."""
        config = self._get_required_config(
            {"train_tokenizer_config": self.train_tokenizer_config},
            "train_tokenizer_config",
            "--train_tokenizer_config is required at iteration > 0.",
        )
        return Path(config.exp_dir) / f"beats_tokenizer_iter{self.iteration}.pt"

    # ---------------------------------------------------------
    # Stages
    # ---------------------------------------------------------
    def train_tokenizer(self, *args, **kwargs):
        """Train the BEATs VQ tokenizer for iterations greater than 0.

        The tokenizer is distilled from the teacher encoder at
        ``train_tokenizer_config.model.beats_teacher_ckpt_path`` (normally
        ``beats_encoder_iter<N-1>.pt``) and exported to
        ``<train_tokenizer_config.exp_dir>/beats_tokenizer_iter<N>.pt``.
        Re-running is safe: the stage is skipped when that file exists.

        Raises:
            RuntimeError: If ``train_tokenizer_config`` is missing.
            FileNotFoundError: If the teacher checkpoint does not exist.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)
        if self.iteration == 0:
            logger.info(
                "Iteration 0 uses the random-projection tokenizer; "
                "skipping train_tokenizer()."
            )
            return None

        output_path = self.get_tokenizer_checkpoint_path()
        if output_path.exists():
            logger.info("Tokenizer already exported: %s. Skipping.", output_path)
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
                f"for iteration {self.iteration - 1} first."
            )
        logger.info(
            "Training BEATs tokenizer | iteration=%d teacher=%s exp_dir=%s",
            self.iteration,
            teacher,
            config.exp_dir,
        )
        trainer = run_training(config)
        return self._export_on_rank_zero(config.exp_dir, output_path, trainer)

    def infer(self, *args, **kwargs):
        """Tokenize the configured test sets into BEATs training targets.

        Writes ``<inference_dir>/<test_name>/target.scp`` (``<idx> <ids...>``)
        and ``<inference_dir>/<test_name>/target_shape`` (``<idx> <length>``,
        used as a batching shape file) for every ``inference_config.dataset.test``
        entry. At iteration > 0, ``inference_config.model.tokenizer_ckpt_path``
        defaults to this iteration's exported tokenizer.

        Raises:
            ValueError: If ``training_config.target_dir`` and
                ``inference_config.inference_dir`` differ.
            FileNotFoundError: If the tokenizer checkpoint is missing.
        """
        self._reject_stage_args("infer", args, kwargs)
        config = self.inference_config
        self._validate_target_dir()
        model_config = config.model
        if self.iteration > 0 and not model_config.get("tokenizer_ckpt_path"):
            model_config.tokenizer_ckpt_path = str(self.get_tokenizer_checkpoint_path())
        tokenizer_ckpt_path = model_config.get("tokenizer_ckpt_path")
        if tokenizer_ckpt_path and not Path(tokenizer_ckpt_path).is_file():
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found: {tokenizer_ckpt_path}. Run the "
                f"`train_tokenizer` stage for iteration {self.iteration} first."
            )
        logger.info(
            "Tokenizing with %s tokenizer | iteration=%d",
            tokenizer_ckpt_path or "random-projection",
            self.iteration,
        )
        result = super().infer()
        for test_set in config.dataset.test:
            test_dir = Path(config.inference_dir) / test_set.name
            write_target_shape(test_dir / "target.scp", test_dir / "target_shape")
        return result

    def collect_stats(self, *args, **kwargs):
        """Write the token list if needed, then collect shape statistics."""
        self._reject_stage_args("collect_stats", args, kwargs)
        self._ensure_token_list()
        return super().collect_stats()

    def train(self, *args, **kwargs):
        """Train the BEATs encoder and export ``beats_encoder_iter<N>.pt``."""
        self._reject_stage_args("train", args, kwargs)
        self._ensure_token_list()
        trainer = super().train()
        return self._export_on_rank_zero(
            self.training_config.exp_dir,
            self.get_encoder_checkpoint_path(),
            trainer,
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _export_on_rank_zero(exp_dir, output_path, trainer) -> Path | None:
        # Lightning's DDP launcher runs this stage in every rank; only the
        # global rank 0 writes the exported checkpoint.
        if rank_zero_only.rank != 0:
            return None
        return export_beats_checkpoint(exp_dir, output_path, trainer=trainer)

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
        write_token_list(token_list, codebook_size)


def write_token_list(output_path: str | Path, codebook_size: int) -> Path:
    """Write the BEATs token list: ``<unk>`` followed by ``0 .. codebook_size-1``.

    The ``<unk>`` entry keeps ids 1-based after ``CommonPreprocessor``
    word tokenization; ``BeatsPretrainModel`` subtracts 1 before the loss.
    An existing file is kept when it has the expected content.

    Args:
        output_path: Token list file to write.
        codebook_size: Number of tokenizer codebook entries.

    Returns:
        Path: ``output_path``.

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
    tmp_path = output.with_name(output.name + ".tmp")
    tmp_path.write_text("\n".join(expected) + "\n", encoding="utf-8")
    tmp_path.replace(output)
    logger.info("Wrote token list (%d entries): %s", len(expected), output)
    return output


def write_target_shape(target_path: str | Path, output_path: str | Path) -> Path:
    """Write a batching shape file (``<idx> <num_tokens>``) from ``target.scp``.

    Args:
        target_path: Index-keyed ``target.scp`` written by the ``infer`` stage.
        output_path: Shape file to write. It is used together with
            ``collect_stats``' ``feats_shape`` by ESPnet's length batch sampler.

    Returns:
        Path: ``output_path``.
    """
    output = Path(output_path)
    tmp_path = output.with_name(output.name + ".tmp")
    with (
        Path(target_path).open("r", encoding="utf-8") as reader,
        tmp_path.open("w", encoding="utf-8") as writer,
    ):
        for line in reader:
            key, _, tokens = line.rstrip("\n").partition(" ")
            writer.write(f"{key} {len(tokens.split())}\n")
    tmp_path.replace(output)
    return output
