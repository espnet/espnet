"""Prepare recipe LM text before running the ESPnet3 LMSystem."""

import argparse
import gzip
from pathlib import Path

from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.download_utils import download_url


def prepare_lm_text(config):
    """Concatenate source LM text and optional LibriSpeech text, keeping IDs.

    Args:
        config: LM config with ``exp_dir``, ``train_text`` and optional
            ``external_text`` (``archive`` path and download ``url``).

    Returns:
        Path to the nonempty, ID-prefixed LM training text.

    Raises:
        FileNotFoundError: The ASR builder has not produced train_text.
        RuntimeError: No nonempty training lines were found.

    Examples:
        After data preparation:

        >>> config = load_and_merge_config(
        ...     "conf/training_lm.yaml", "training.yaml",
        ...     default_package="egs3.TEMPLATE.esp2_asr",
        ... )
        >>> text_path = prepare_lm_text(config)
        >>> text_path.is_file()
        True
    """
    output = Path(config.exp_dir).resolve() / "lm_train.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as destination:
        with Path(config.train_text).open(encoding="utf-8") as source:
            for line in source:
                if len(line.split()) > 1:
                    destination.write(line.rstrip("\n") + "\n")
                    count += 1
        if config.get("external_text"):
            archive = Path(config.external_text.archive)
            if not archive.is_file():
                part = archive.with_suffix(".part")
                download_url(config.external_text.url, part)
                part.replace(archive)
            with gzip.open(archive, "rt", encoding="utf-8") as source:
                for index, line in enumerate(source, 1):
                    if line.strip():
                        destination.write(
                            f"librispeech_lng_{index:08d} {line.rstrip()}\n"
                        )
                        count += 1
    if not count:
        raise RuntimeError("Language model training text is empty")
    temporary.replace(output)
    return output


def main():
    """Prepare LM text using the same config as the LM training entrypoint.

    Args:
        None. Read --config from the command line.

    Returns:
        None. Write the combined ID-prefixed text under exp_dir.

    Examples:
        From the prepared recipe directory::

            python -m egs3.an4.esp2_asr.src.language_model \
                --config conf/training_lm.yaml
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_and_merge_config(
        args.config, "training.yaml", default_package="egs3.TEMPLATE.esp2_asr"
    )
    prepare_lm_text(config)


if __name__ == "__main__":
    main()
