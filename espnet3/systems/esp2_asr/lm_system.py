"""Train ESPnet2-derived language models with the shared ESPnet3 trainer."""

from pathlib import Path

from hydra.utils import instantiate

from espnet3.systems.base.system import BaseSystem


class LMSystem(BaseSystem):
    """Use the base training stage and collect token lengths for LM batching.

    Supply an ESPnet3 training config with ``task`` set to
    ``espnet3.systems.esp2_asr.lm_task.LMTask``. The dataset uses the ASR
    tokenizer through CommonPreprocessor; the collator must pad text with 0.
    Training, precision, optimization and checkpoint averaging are inherited
    unchanged from BaseSystem. No acoustic normalization statistics are needed.

    Args:
        training_config: ESPnet3 config with dataset, model.token_list,
            stats_dir, dataloader, optimizer and trainer settings.
        inference_config: Optional inherited inference configuration.
        metrics_config: Optional inherited measurement configuration.
        publication_config: Optional inherited publication configuration.
        stage_log_mapping: Optional inherited stage log directory overrides.
        demo_config: Optional inherited demo configuration.

    Examples:
        >>> system = LMSystem(training_config=config)
        >>> system.collect_stats()
        >>> system.train()
    """

    def collect_stats(self, *args, **kwargs):
        """Write token-length shapes using the configured public preprocessor.

        ESPnetLanguageModel.collect_feats returns no acoustic features. This
        LM-specific stage instead writes ``text_shape`` for the existing folded
        or numel sampler. Its second dimension is the vocabulary size, matching
        the source LM recipe's numel convention. IDs are DataOrganizer indices.

        Args:
            *args: Must be empty; settings come from training_config.
            **kwargs: Must be empty; settings come from training_config.

        Returns:
            None. Atomic shape files are written to stats_dir/{train,valid}.

        Raises:
            TypeError: Stage arguments were supplied.
            ValueError: A required split or the vocabulary is empty.
            FileNotFoundError: Text or tokenizer files have not been prepared.

        Examples:
            >>> system = LMSystem(training_config=config)
            >>> system.collect_stats()
            >>> (Path(config.stats_dir) / "train/text_shape").is_file()
            True
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        config = self.training_config
        tokens = config.model.token_list
        if isinstance(tokens, str):
            tokens = Path(tokens).read_text(encoding="utf-8").splitlines()
        if not tokens:
            raise ValueError("LM training requires a nonempty token list")
        organizer = instantiate(config.dataset)
        for split in ("train", "valid"):
            dataset = getattr(organizer, split)
            if dataset is None or len(dataset) == 0:
                raise ValueError(f"LM {split} dataset is empty")
            destination = Path(config.stats_dir) / split / "text_shape"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for index in range(len(dataset)):
                    text = dataset[index]["text"]
                    stream.write(f"{index} {len(text)},{len(tokens)}\n")
            temporary.replace(destination)
