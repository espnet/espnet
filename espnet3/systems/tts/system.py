"""TTS system implementation.

This module adds new stages on top of the base system: removing long-short
utterances and creating token lists, plus the stats-collection and training
stages. Both training stages dispatch to a GAN-specific Lightning trainer
when the configured model is a GAN-TTS model (e.g. VITS).
"""

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.base.training import _ensure_directories, _instantiate_model
from espnet3.systems.tts.gan_trainer import build_gan_trainer
from espnet3.systems.tts.remove_long_short_provider import RemoveLongShortProvider
from espnet3.systems.tts.remove_long_short_runner import RemoveLongShortRunner
from espnet3.systems.tts.xvector_provider import XVectorProvider
from espnet3.systems.tts.xvector_runner import XVectorRunner
from espnet3.utils.task_utils import save_espnet_config

logger = logging.getLogger(__name__)


def _build_trainer(config: DictConfig) -> ESPnet3LightningTrainer:
    """Build the Lightning trainer for a TTS config.

    Shadows ``espnet3.systems.base.training._build_trainer`` to add GAN-TTS
    dispatch: GAN-TTS models (VITS, JETS, ...) need a second optimizer and a
    generator/discriminator step schedule, which the plain
    ``ESPnetLightningModule``/``ESPnet3LightningTrainer`` pair cannot express.
    The non-GAN branch is deliberately a copy of the base builder's body rather
    than a delegation to it, so that the model is instantiated exactly once -
    instantiating and discarding a model would advance the global RNG and make
    training depend on whether this dispatch happened.

    Args:
        config (DictConfig): The training config. ``config.model`` (with
            ``config.task``, if set) selects the model; ``config.trainer``,
            ``config.exp_dir`` and ``config.best_model_criterion`` configure
            the trainer.

    Returns:
        ESPnet3LightningTrainer: A ``GANTTSLightningTrainer`` if the model is
        an ``AbsGANESPnetModel``, otherwise a plain ``ESPnet3LightningTrainer``.
    """
    model = _instantiate_model(config)
    if isinstance(model, AbsGANESPnetModel):
        return build_gan_trainer(config, model)

    lit_model = ESPnetLightningModule(model, config)
    return ESPnet3LightningTrainer(
        model=lit_model,
        exp_dir=config.exp_dir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )


class TTSSystem(BaseSystem):
    """TTS-specific system.

    This system adds:
      - Removing long-short utterances
      - Creating token lists

    Additional stage log paths:
        | Stage                 | Path reference                  |
        |---                   |---                              |
        | compute_xvectors     | training_config.xvector.save_path |
        | remove_long_short    | training_config.remove_long_short.save_path |
        | create_token_list    | training_config.create_token_list.save_path |
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        stage_log_mapping=None,
        **kwargs,
    ) -> None:
        """Initialize the TTS system with TTS-specific stage mappings.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            stage_log_mapping (dict | None): Extra per-stage log path
                overrides contributed by a subclass. Merged on top of the two
                TTS stages registered here rather than replacing them, so a
                recipe-local subclass can register its own stages (e.g.
                ``compute_xvectors``) without having to restate these.
            **kwargs: Forwarded to :class:`BaseSystem`.
        """
        mapping = {
            "compute_xvectors": "training_config.xvector.save_path",
            "remove_long_short": "training_config.remove_long_short.save_path",
            "create_token_list": "training_config.create_token_list.save_path",
        }
        if stage_log_mapping:
            mapping.update(stage_log_mapping)
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            stage_log_mapping=mapping,
            **kwargs,
        )

    def compute_xvectors(self, *args, **kwargs):
        r"""Compute x-vectors for multiple data splits using parallel execution.

        X-vectors (speaker embeddings) are extracted using a pre-trained
        model for train, valid, and test splits. They can be used as
        speaker conditioning in TTS models.

        This method uses espnet3 manifest files generated by the dataset
        builder. Manifest format: ``utt_id\twav_path\ttext\tspeaker_id``
        (TSV).

        Args:
            *args: Must be empty. Passing any positional argument raises
                ``RuntimeError`` via ``_reject_stage_args``.
            **kwargs: Must be empty. Passing any keyword argument raises
                ``RuntimeError`` via ``_reject_stage_args``.

        Returns:
            None.

        Raises:
            RuntimeError: If required configuration is missing or
                manifest files are not found.

        Notes:
            Configuration should include:
                training_config.xvector.pretrained_model: Model tag or path.
                training_config.xvector.toolkit: ``espnet``, ``speechbrain``,
                    or ``rawnet``.
                training_config.xvector.save_path: Output directory.
                training_config.xvector.splits: Splits to process
                    (train, valid, test).
                training_config.xvector.batch_size: Batch size for
                    processing.
                training_config.xvector.device: Device to use (default:
                    ``cuda:0`` if available).

        Examples:
            ```bash
            python run.py --stages compute_xvectors \
                --training_config conf/training.yaml
            ```
            writes one embedding per utterance:
            ```text
            data/x_vectors/spkrec-ecapa-voxceleb_train/19_198_000000_000000.pt
            ```
        """
        self._reject_stage_args("compute_xvectors", args, kwargs)
        logger.info("TTSSystem.compute_xvectors(): starting x-vector computation")

        # Parse the parallel config early so it applies to the x-vector runner.
        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        xvec_cfg = self.training_config.get("xvector", None)
        if xvec_cfg is None:
            raise RuntimeError(
                "training_config.xvector must be set for compute_xvectors stage."
            )
        save_path_str = xvec_cfg.get("save_path", None)
        if save_path_str is None:
            raise RuntimeError(
                "training_config.xvector.save_path must be set for "
                "compute_xvectors stage."
            )
        save_path = Path(save_path_str)
        save_path.mkdir(parents=True, exist_ok=True)

        # Get list of splits to process (Default: all splits)
        splits = xvec_cfg.get("splits", ["train", "valid", "test"])

        if isinstance(splits, str):
            splits = [splits]

        manifest_paths = xvec_cfg.get("manifest_paths", {})

        logger.info(f"Will process splits: {splits}")
        logger.info(f"Manifest paths: {manifest_paths}")

        # Validate toolkit and model
        toolkit = xvec_cfg.get("toolkit", "speechbrain")
        pretrained_model = xvec_cfg.get(
            "pretrained_model", "speechbrain/spkrec-ecapa-voxceleb"
        )
        device = xvec_cfg.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Process each split
        for split in splits:
            logger.info(f"Processing split: {split}")

            manifest_path = manifest_paths.get(split, None)
            if manifest_path is None:
                manifest_path = f"data/manifest/{split}.tsv"
            manifest_path = Path(manifest_path).resolve()
            if not manifest_path.exists():
                raise RuntimeError(
                    f"Manifest file not found for split '{split}': "
                    f"{manifest_path}. Please generate the manifest file "
                    "using the create_dataset stage and ensure the path "
                    "is correct."
                )

            utterances, _ = XVectorProvider._load_manifest(manifest_path)
            n_utts = len(utterances)
            if n_utts == 0:
                raise RuntimeError(f"No utterances found in manifest: {manifest_path}.")

            logger.info(f"Split '{split}': {n_utts} utterances in {manifest_path}")

            batch_size = xvec_cfg.get("batch_size", None)
            async_mode = xvec_cfg.get("async_mode", False)
            spk_embed_tag = xvec_cfg.get("spk_embed_tag", "spk_embed")
            output_subdir = save_path / f"{spk_embed_tag}_{split}"

            provider = XVectorProvider(
                config=self.training_config,
                params={
                    "toolkit": toolkit,
                    "pretrained_model": pretrained_model,
                    "device": device,
                    "manifest_path": str(manifest_path),
                    "output_dir": str(output_subdir),
                },
            )

            runner = XVectorRunner(
                provider=provider,
                batch_size=batch_size,
                async_mode=async_mode,
            )

            logger.info(
                f"Processing {n_utts} utterances for x-vector extraction "
                f"(split: {split})"
            )

            indices = list(range(n_utts))
            results = runner(indices)

            if results is None:
                logger.info(
                    f"Async job submitted for split '{split}'. Check "
                    "result directory for outputs."
                )
                continue

            flat = []
            for item in results:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            n_ok = sum(1 for r in flat if r.get("status") == "ok")
            n_skipped = sum(1 for r in flat if r.get("status") == "skipped")
            logger.info(
                f"X-vectors for split '{split}' saved to {output_subdir} "
                f"({n_ok} new, {n_skipped} skipped)"
            )

        logger.info("X-vector computation completed for all splits")

    def remove_long_short(self, *args, **kwargs):
        """Remove long-short utterances based on duration thresholds.

        This stage processes manifest files to filter out utterances that
        are too short or too long based on specified duration thresholds.
        It reads WAV headers via soundfile to check audio durations (in
        parallel, via RemoveLongShortProvider/Runner) and saves filtered
        manifests for downstream stages.

        Configuration should include (under
        ``training_config.remove_long_short``):
            - ``min_wav_duration``: Minimum duration in seconds
            - ``max_wav_duration``: Maximum duration in seconds
            - ``save_path``: Directory to save filtered manifests
            - ``splits``: List of splits to process (train, valid, test)
            - ``manifest_paths``: Optional dict of split to manifest path
              (default: data/manifest/{split}.tsv)

        Example:
            .. code-block:: yaml

                remove_long_short:
                  min_wav_duration: 2
                  max_wav_duration: 30
                  save_path: data/manifest_filtered
                  splits: [train, valid, test]

        Raises:
            RuntimeError: If required configuration is missing or manifest
                files not found.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info(
            "TTSSystem.remove_long_short(): starting long-short utterance removal"
        )

        remove_long_short_config = self._get_required_config(
            self.training_config,
            "remove_long_short",
            "training_config.remove_long_short must be set for "
            "remove_long_short stage.",
        )
        save_path = Path(
            self._get_required_config(
                remove_long_short_config,
                "save_path",
                "training_config.remove_long_short.save_path must be set "
                "for remove_long_short stage.",
            )
        )

        duration_error = (
            "training_config.remove_long_short.min_wav_duration and "
            "max_wav_duration must be set for remove_long_short stage."
        )
        min_duration = self._get_required_config(
            remove_long_short_config, "min_wav_duration", duration_error
        )
        max_duration = self._get_required_config(
            remove_long_short_config, "max_wav_duration", duration_error
        )

        # Parse the parallel configuration early to set up parallelism for
        # the duration-filtering runner.
        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        # Get list of splits to process (Default: all splits)
        splits = remove_long_short_config.get("splits", ["train", "valid", "test"])

        if isinstance(splits, str):
            splits = [splits]

        manifest_paths = remove_long_short_config.get("manifest_paths", {})
        batch_size = remove_long_short_config.get("batch_size", None)

        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Removing long-short utterances with "
            f"min_duration={min_duration}s, max_duration={max_duration}s"
        )

        for split in splits:
            logger.info(f"Processing split: {split}")

            manifest_path = manifest_paths.get(split) if manifest_paths else None
            if manifest_path is None:
                manifest_path = f"data/manifest/{split}.tsv"
            manifest_path = Path(manifest_path).resolve()
            filtered_manifest_path = save_path / manifest_path.name
            if not manifest_path.exists():
                raise RuntimeError(
                    f"Manifest file not found for split '{split}': "
                    f"{manifest_path}. Please generate the manifest file using "
                    "the create_dataset stage and ensure the path is correct."
                )

            entries, n_dropped_empty = RemoveLongShortProvider._load_entries(
                manifest_path
            )
            n_entries = len(entries)

            provider = RemoveLongShortProvider(
                config=self.training_config,
                params={
                    "manifest_path": str(manifest_path),
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                },
            )

            # resume=False: the keep/drop decisions depend on the duration
            # bounds, so shard results from an earlier run (possibly with
            # different bounds) must never be reused.
            runner = RemoveLongShortRunner(
                provider=provider,
                batch_size=batch_size,
                output_dir=save_path / "shards",
                shard_subdir=split,
                resume=False,
            )

            logger.info(
                f"Checking durations for {n_entries} utterances (split: {split})"
            )

            indices = list(range(n_entries))
            # merge() returns the shard records flattened and re-sorted by idx.
            results = runner(indices) if n_entries else []
            keep_by_idx = {r["idx"]: r["keep"] for r in results}

            n_kept = 0
            n_dropped_duration = 0
            filtered_entries = []
            for idx, (_, _, line) in enumerate(entries):
                if keep_by_idx[idx]:
                    filtered_entries.append(line)
                    n_kept += 1
                else:
                    n_dropped_duration += 1

            with open(filtered_manifest_path, "w", encoding="utf-8") as f:
                f.writelines(filtered_entries)

            logger.info(
                f"Split '{split}': kept {n_kept}, dropped {n_dropped_duration} "
                f"by duration, dropped {n_dropped_empty} by empty text → "
                f"{filtered_manifest_path}"
            )

        logger.info(
            "Long-short utterance removal completed. Filtered manifests "
            f"saved to: {save_path}"
        )

    def create_token_list(self, *args, **kwargs):
        """Create token list from training data.

        This stage processes the training manifest to extract unique
        tokens from the text transcriptions and saves them to a token
        list file.

        Configuration should include (under
        ``training_config.create_token_list``):
            - ``save_path``: Directory to save the token list file
            - ``filename``: Token list file name (e.g. tokens.txt)
            - ``manifest_path``: Path to the training manifest file
              (default: data/manifest/train.tsv)
            - ``token_type``: Tokenization type such as char, word, bpe,
              or phn (default: char)
            - ``cleaner``: Optional text cleaner name (e.g. tacotron)
            - ``g2p``: Optional grapheme-to-phoneme model name
            - ``add_symbol`` / ``add_nonsplit_symbol``: Special symbols
              to insert, as "<symbol>:<index>" strings
            - ``cutoff`` / ``vocabulary_size``: Frequency cutoff and
              vocabulary size limit (default: 0 = unlimited)
            - ``vocab_builder`` / ``vocab_builder_conf``: Optional custom
              vocab builder callable path and its options; when set it
              replaces the default frequency-count construction

        Example:
            .. code-block:: yaml

                create_token_list:
                  save_path: data/token_list
                  filename: tokens.txt
                  manifest_path: data/manifest/train.tsv
                  token_type: phn
                  cleaner: tacotron
                  g2p: g2p_en
                  add_symbol:
                    - "<blank>:0"
                    - "<unk>:1"
                    - "<sos/eos>:-1"

        Raises:
            RuntimeError: If required configuration is missing or
                manifest file not found.
        """
        self._reject_stage_args("create_token_list", args, kwargs)
        logger.info("TTSSystem.create_token_list(): starting token list creation")

        create_token_list_config = self._get_required_config(
            self.training_config,
            "create_token_list",
            "training_config.create_token_list must be set for "
            "create_token_list stage.",
        )
        save_path_str = self._get_required_config(
            create_token_list_config,
            "save_path",
            "training_config.create_token_list.save_path must be set "
            "for create_token_list stage.",
        )
        filename = self._get_required_config(
            create_token_list_config,
            "filename",
            "training_config.create_token_list.filename must be set "
            "(e.g. 'tokens.txt'); save_path is the output directory.",
        )
        save_dir = Path(save_path_str)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / filename

        manifest_path = Path(
            create_token_list_config.get("manifest_path", "data/manifest/train.tsv")
        ).resolve()

        if not manifest_path.exists():
            raise RuntimeError(
                f"Manifest file not found for token list creation: "
                f"{manifest_path}. Please ensure the manifest file is "
                "generated and the path is correct."
            )

        # Optional custom vocab builder (e.g. F5 pinyin, prepare_emilia-style):
        # a callable ``fn(texts: list[str], **conf) -> list[str]`` returning the
        # full ordered token list. When set it fully replaces the default
        # frequency-count + special-symbol construction below, so any dataset /
        # tokenizer can plug its own vocab construction into this stage.
        vocab_builder_path = create_token_list_config.get("vocab_builder", None)
        if vocab_builder_path is not None:
            from hydra.utils import get_method

            cleaner_fn = TextCleaner(create_token_list_config.get("cleaner", None))
            texts = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) > 2 and parts[2].strip():
                        texts.append(cleaner_fn(parts[2]))

            builder = get_method(path=vocab_builder_path)
            builder_conf = create_token_list_config.get("vocab_builder_conf", {}) or {}
            if not isinstance(builder_conf, dict):
                builder_conf = OmegaConf.to_container(builder_conf, resolve=True)
            tokens = builder(texts, **builder_conf)

            with open(output_file, "w", encoding="utf-8") as f:
                for tok in tokens:
                    f.write(f"{tok}\n")
            logger.info(
                "create_token_list: built %d tokens from %d transcripts via %s -> %s",
                len(tokens),
                len(texts),
                vocab_builder_path,
                output_file,
            )
            return

        # Declare text processing parameters with defaults (matching
        # espnet2 defaults where applicable)
        cleaner = create_token_list_config.get("cleaner", None)
        token_type = create_token_list_config.get("token_type", "char")
        bpemodel = create_token_list_config.get("bpemodel", None)
        delimiter = create_token_list_config.get("delimiter", None)
        space_symbol = create_token_list_config.get("space_symbol", "<space>")
        non_linguistic_symbols = create_token_list_config.get(
            "non_linguistic_symbols", None
        )
        remove_non_linguistic_symbols = create_token_list_config.get(
            "remove_non_linguistic_symbols", False
        )
        g2p = create_token_list_config.get("g2p", None)
        add_symbol = create_token_list_config.get("add_symbol", [])
        add_nonsplit_symbol = create_token_list_config.get("add_nonsplit_symbol", [])
        cutoff = create_token_list_config.get("cutoff", 0)
        vocabulary_size = create_token_list_config.get("vocabulary_size", 0)

        cleaner: TextCleaner = TextCleaner(cleaner)
        tokenizer = build_tokenizer(
            token_type=token_type,
            bpemodel=bpemodel,
            delimiter=delimiter,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
            g2p_type=g2p,
            nonsplit_symbol=add_nonsplit_symbol,
        )

        counter = Counter()

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                parts = line.split("\t")
                text = parts[2]
                cleaned_text = cleaner(text)
                tokens = tokenizer.text2tokens(cleaned_text)
                for t in tokens:
                    counter[t] += 1

        # Sort by the number of occurrences in descending order
        # and filter lower frequency words than cutoff value
        words_and_counts = list(
            filter(
                lambda x: x[1] > cutoff, sorted(counter.items(), key=lambda x: -x[1])
            )
        )

        # Restrict the vocabulary size
        if vocabulary_size > 0:
            if vocabulary_size < len(add_symbol):
                raise RuntimeError(f"vocabulary_size is too small: {vocabulary_size}")
            words_and_counts = words_and_counts[: vocabulary_size - len(add_symbol)]

        # Parse the values of --add_symbol and --add_nonsplit_symbol
        for symbol_and_id in add_symbol + add_nonsplit_symbol:
            # e.g symbol="<blank>:0"
            try:
                symbol, idx = symbol_and_id.split(":")
                idx = int(idx)
            except ValueError:
                raise RuntimeError(f"Format error: e.g. '<blank>:0': {symbol_and_id}")
            symbol = symbol.strip()

            # e.g. idx=0  -> append as the first symbol
            # e.g. idx=-1 -> append as the last symbol
            if idx < 0:
                idx = len(words_and_counts) + 1 + idx
            words_and_counts.insert(idx, (symbol, None))

        # Write words
        with open(output_file, "w", encoding="utf-8") as f:
            for word, count in words_and_counts:
                f.write(f"{word}\n")

        # Logging
        total_count = sum(counter.values())
        invocab_count = sum(c for w, c in words_and_counts if c is not None)
        if total_count > 0:
            logger.info(
                f"OOV rate = {(total_count - invocab_count) / total_count * 100:.2f} %"
            )
        else:
            logger.warning("create_token_list: manifest contained no tokens.")

    def _prepare_training_runtime(self) -> None:
        """Set up directories, parallelism, seeding and matmul precision.

        The base training module inlines this same sequence inside its
        ``collect_stats`` / ``train`` functions instead of exposing it as a
        reusable helper, so there is nothing to delegate to beyond
        ``_ensure_directories``.
        """
        config = self.training_config
        _ensure_directories(config)

        if config.get("parallel"):
            set_parallel(config.parallel)

        if config.get("seed") is not None:
            L.seed_everything(int(config.seed), workers=True)

        torch.set_float32_matmul_precision("high")

    def collect_stats(self, *args, **kwargs):
        """Run the collect_stats stage using the configured trainer.

        Prepares the training runtime (directories, parallelism, seed), then
        delegates to the trainer's ``collect_stats`` method.  Positional and
        keyword stage arguments are rejected to avoid silent misconfiguration.

        This override exists solely so that ``model.normalize`` /
        ``model.normalize_conf`` survive into the trainer.  It is load-bearing;
        see the Notes below before replacing it with the inherited stage.

        Args:
            *args: Must be empty.  Passing any positional argument raises
                ``TypeError`` via ``_reject_stage_args``.
            **kwargs: Must be empty.  Passing any keyword argument raises
                ``TypeError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            TypeError: If any positional or keyword arguments are passed.

        Notes:
            Do not delete this override in favour of
            ``espnet3.systems.base.training.collect_stats``.

            The base implementation pops ``normalize`` and ``normalize_conf``
            out of ``config.model`` before building the trainer.  For a task
            whose normalizer is optional that is harmless, but for TTS it is
            not: ``espnet2.tasks.tts`` declares ``normalize_choices`` with
            ``default="global_mvn"``.  Removing the key therefore does not mean
            "no normalizer" - it restores the ``global_mvn`` default.

            The F5-TTS recipe configs set ``normalize: null`` on purpose, and
            that ``null`` has to reach the task builder intact.  If the key is
            popped, the task builds a ``GlobalMVN`` and stats collection dies
            with::

                GlobalMVN.__init__() missing 1 required positional argument:
                'stats_file'

            which is a chicken-and-egg failure: the stats file GlobalMVN wants
            is exactly the artifact this stage is being run to produce.

            Note that the base ``train()`` does *not* pop these keys - only the
            base ``collect_stats`` does.  That asymmetry is why ``train`` is
            safely inherited from ``BaseSystem`` while ``collect_stats`` is not.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({"exp_dir": "/tmp/exp"})
            >>> system = TTSSystem(training_config=cfg)
            >>> system.collect_stats()  # runs stats collection end-to-end
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        # Build the trainer WITHOUT popping normalize/normalize_conf, unlike
        # the base collect_stats. See the Notes in this docstring.
        trainer = _build_trainer(self.training_config)
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            getattr(self.training_config, "stats_dir", None),
        )

    def train(self, *args, **kwargs):
        """Run the training stage using the configured trainer.

        Mirrors :func:`espnet3.systems.base.training.train` exactly, except
        that the trainer comes from this module's GAN-aware
        :func:`_build_trainer`. The override exists only for that dispatch:
        the base implementation resolves ``_build_trainer`` in its own module
        namespace, so a GAN-TTS model would otherwise be wrapped in the plain
        single-optimizer trainer and fail at the first discriminator step.

        Args:
            *args: Must be empty. Passing any positional argument raises via
                ``_reject_stage_args``.
            **kwargs: Must be empty. Passing any keyword argument raises via
                ``_reject_stage_args``.

        Returns:
            None

        Notes:
            ``training_config.fit`` is forwarded verbatim to ``trainer.fit``.
        """
        self._reject_stage_args("train", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        task = self.training_config.get("task")
        if task:
            save_espnet_config(task, self.training_config, self.training_config.exp_dir)

        trainer = _build_trainer(self.training_config)

        fit_kwargs: Dict[str, Any] = {}
        if hasattr(self.training_config, "fit") and self.training_config.fit:
            fit_kwargs = OmegaConf.to_container(self.training_config.fit, resolve=True)

        trainer.fit(**fit_kwargs)
        logger.info(
            "Training finished in %.2fs | exp_dir=%s model=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            (
                self.training_config.model.get("_target_", None)
                if isinstance(self.training_config.model, DictConfig)
                else None
            ),
        )
