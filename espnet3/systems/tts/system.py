"""TTS system implementation.

This module adds new stages on top of the base system: removing long-short
utterances and creating token lists, plus the stats-collection stage.
"""

import logging
import time
from collections import Counter
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem

from espnet3.systems.base.training import _build_trainer, _ensure_directories
from espnet3.systems.tts.remove_long_short_provider import RemoveLongShortProvider
from espnet3.systems.tts.remove_long_short_runner import RemoveLongShortRunner

logger = logging.getLogger(__name__)


class TTSSystem(BaseSystem):
    """TTS-specific system.

    This system adds:
      - Removing long-short utterances
      - Creating token lists

    Additional stage log paths:
        | Stage                 | Path reference                  |
        |---                   |---                              |
        | remove_long_short    | training_config.remove_long_short.save_path |
        | create_token_list    | training_config.create_token_list.save_path |
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        """Initialize the TTS system with TTS-specific stage mappings."""
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            stage_log_mapping={
                "remove_long_short": "training_config.remove_long_short.save_path",
                "create_token_list": "training_config.create_token_list.save_path",
            },
            **kwargs,
        )

    def remove_long_short(self, *args, **kwargs):
        """Remove long-short utterances based on duration thresholds.

        This stage processes manifest files to filter out utterances that
        are too short or too long based on specified duration thresholds.
        It reads WAV headers via soundfile to check audio durations (in
        parallel, via RemoveLongShortProvider/Runner) and saves filtered
        manifests for downstream stages.

        Configuration should include:
            training_config.remove_long_short.min_wav_duration: Minimum
                duration in seconds
            training_config.remove_long_short.max_wav_duration: Maximum
                duration in seconds
            training_config.remove_long_short.save_path: Directory to save
                filtered manifests
            training_config.remove_long_short.splits: List of splits to
                process (train, valid, test)
            training_config.remove_long_short.manifest_paths: Optional dict
                of split to manifest path (default: data/manifest/{split}.tsv)

        Raises:
            RuntimeError: If required configuration is missing or manifest
                files not found.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info(
            "TTSSystem.remove_long_short(): starting long-short utterance removal"
        )

        rls_cfg = self.training_config.get("remove_long_short", None)
        if rls_cfg is None:
            raise RuntimeError(
                "training_config.remove_long_short must be set for "
                "remove_long_short stage."
            )
        save_path_str = rls_cfg.get("save_path", None)
        if save_path_str is None:
            raise RuntimeError(
                "training_config.remove_long_short.save_path must be set "
                "for remove_long_short stage."
            )
        save_path = Path(save_path_str)

        min_duration = rls_cfg.get("min_wav_duration", None)
        max_duration = rls_cfg.get("max_wav_duration", None)
        if min_duration is None or max_duration is None:
            raise RuntimeError(
                "training_config.remove_long_short.min_wav_duration and "
                "max_wav_duration must be set for remove_long_short stage."
            )

        # Parse the parallel configuration early to set up parallelism for
        # the duration-filtering runner.
        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        # Get list of splits to process (Default: all splits)
        splits = rls_cfg.get("splits", ["train", "valid", "test"])

        if isinstance(splits, str):
            splits = [splits]

        manifest_paths = rls_cfg.get("manifest_paths", {})
        batch_size = rls_cfg.get("batch_size", None)
        async_mode = rls_cfg.get("async_mode", False)

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

            runner = RemoveLongShortRunner(
                provider=provider,
                batch_size=batch_size,
                async_mode=async_mode,
            )

            logger.info(
                f"Checking durations for {n_entries} utterances (split: {split})"
            )

            indices = list(range(n_entries))
            results = runner(indices) if n_entries else []

            if results is None:
                logger.info(
                    f"Async job submitted for split '{split}'. Check result "
                    "directory for outputs."
                )
                continue

            flat = []
            for item in results:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)

            # parallel_for yields results in completion order, not submission
            # order, so re-sort by idx to keep the manifest deterministic.
            flat.sort(key=lambda r: r["idx"])
            keep_by_idx = {r["idx"]: r["keep"] for r in flat}

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

        Configuration should include:
            training_config.create_token_list.save_path: Path to save
                the token list file
            training_config.create_token_list.manifest_path: Path to the
                training manifest file (default: data/manifest/train.tsv)

        Raises:
            RuntimeError: If required configuration is missing or
                manifest file not found.
        """
        self._reject_stage_args("create_token_list", args, kwargs)
        logger.info("TTSSystem.create_token_list(): starting token list creation")

        tl_cfg = self.training_config.get("create_token_list", None)
        if tl_cfg is None:
            raise RuntimeError(
                "training_config.create_token_list must be set for "
                "create_token_list stage."
            )
        save_path_str = tl_cfg.get("save_path", None)
        if save_path_str is None:
            raise RuntimeError(
                "training_config.create_token_list.save_path must be set "
                "for create_token_list stage."
            )
        filename = tl_cfg.get("filename", None)
        if filename is None:
            raise RuntimeError(
                "training_config.create_token_list.filename must be set "
                "(e.g. 'tokens.txt'); save_path is the output directory."
            )
        save_dir = Path(save_path_str)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / filename

        manifest_path = Path(
            tl_cfg.get("manifest_path", "data/manifest/train.tsv")
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
        vocab_builder_path = tl_cfg.get("vocab_builder", None)
        if vocab_builder_path is not None:
            from hydra.utils import get_method

            cleaner_fn = TextCleaner(tl_cfg.get("cleaner", None))
            texts = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) > 2 and parts[2].strip():
                        texts.append(cleaner_fn(parts[2]))

            builder = get_method(path=vocab_builder_path)
            builder_conf = tl_cfg.get("vocab_builder_conf", {}) or {}
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
        cleaner = tl_cfg.get("cleaner", None)
        token_type = tl_cfg.get("token_type", "char")
        bpemodel = tl_cfg.get("bpemodel", None)
        delimiter = tl_cfg.get("delimiter", None)
        space_symbol = tl_cfg.get("space_symbol", "<space>")
        non_linguistic_symbols = tl_cfg.get("non_linguistic_symbols", None)
        remove_non_linguistic_symbols = tl_cfg.get(
            "remove_non_linguistic_symbols", False
        )
        g2p = tl_cfg.get("g2p", None)
        add_symbol = tl_cfg.get("add_symbol", [])
        add_nonsplit_symbol = tl_cfg.get("add_nonsplit_symbol", [])
        cutoff = tl_cfg.get("cutoff", 0)
        vocabulary_size = tl_cfg.get("vocabulary_size", 0)

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
