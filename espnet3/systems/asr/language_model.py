"""Train the original ESPnet2 LM for an ESPnet3 ASR recipe.

The current ASRSystem has no LM training stage. This explicit compatibility
entrypoint uses the existing LMTask for statistics, training and perplexity;
it does not reimplement the language model or silently disable shallow fusion.
"""

import argparse
import gzip
import json
import subprocess
import sys
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_config_with_defaults
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
    """
    output = Path(config.exp_dir) / "lm_train.txt"
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


def train_language_model(config):
    """Run native LM statistics, training, checkpoint averaging and perplexity.

    Args:
        config: Resolved LM YAML containing native_config, tokenizer_dir,
            exp_dir, train_text, valid_text, test_text, and ngpu. Optional
            native_overrides are explicit experiment-only native task overrides.
            An external_text mapping enables the source VOiCES LM text download.

    Returns:
        Path to the native ``valid.loss.ave.pth`` checkpoint for Speech2Text.

    Raises:
        subprocess.CalledProcessError: A native stage failed; inference must not
            silently continue without the LM.
        FileNotFoundError: A required input or final checkpoint is missing.

    Examples:
        From a prepared recipe directory, run
        ``python -m espnet3.systems.asr.language_model`` with
        ``--config conf/language_model.yaml``.
        The ASR tokenizer must already exist. Use a new exp_dir for new conditions;
        native training resumes its checkpoint when rerunning the same config.
    """
    output = Path(config.exp_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    text = prepare_lm_text(config)
    native = OmegaConf.merge(
        OmegaConf.load(config.native_config), config.get("native_overrides", {})
    )
    native_path = output / "native.yaml"
    OmegaConf.save(native, native_path)
    tokenizer = Path(config.tokenizer_dir).resolve()
    common = [
        sys.executable,
        "-m",
        "espnet2.bin.lm_train",
        "--config",
        str(native_path),
        "--use_preprocessor",
        "true",
        "--token_type",
        "bpe",
        "--token_list",
        str(tokenizer / "tokens.txt"),
        "--bpemodel",
        str(tokenizer / "unigram.model"),
        "--train_data_path_and_name_and_type",
        f"{text},text,text",
        "--valid_data_path_and_name_and_type",
        f"{config.valid_text},text,text",
    ]
    stats = output / "stats"
    commands = []

    def run(arguments):
        commands.append(arguments)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        subprocess.run(arguments, check=True)

    run(common + ["--collect_stats", "true", "--ngpu", "0", "--output_dir", str(stats)])
    vocabulary = len((tokenizer / "tokens.txt").read_text().splitlines())
    for split in ("train", "valid"):
        shape = stats / split / "text_shape"
        shape.with_suffix(".bpe").write_text(
            "".join(f"{line},{vocabulary}\n" for line in shape.read_text().splitlines())
        )
    arguments = common + [
        "--ngpu",
        str(config.ngpu),
        "--fold_length",
        "150",
        "--train_shape_file",
        str(stats / "train/text_shape.bpe"),
        "--valid_shape_file",
        str(stats / "valid/text_shape.bpe"),
        "--resume",
        "true",
        "--output_dir",
        str(output),
    ]
    rendezvous = None
    if int(config.ngpu) > 1:
        # FileStore requires a fresh path, including when resuming a checkpoint.
        rendezvous = output / f"distributed_init.{uuid.uuid4().hex}"
        arguments += [
            "--multiprocessing_distributed",
            "true",
            "--dist_init_method",
            rendezvous.as_uri(),
            "--dist_world_size",
            "1",
        ]
    try:
        run(arguments)
    finally:
        if rendezvous is not None:
            rendezvous.unlink(missing_ok=True)
    checkpoint = output / "valid.loss.ave.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run(
        [
            sys.executable,
            "-m",
            "espnet2.bin.lm_calc_perplexity",
            "--ngpu",
            str(min(int(config.ngpu), 1)),
            "--data_path_and_name_and_type",
            f"{config.test_text},text,text",
            "--train_config",
            str(output / "config.yaml"),
            "--model_file",
            str(checkpoint),
            "--output_dir",
            str(output / "perplexity_test"),
        ]
    )
    return checkpoint


def main():
    """Read one explicit LM config and run the compatibility pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train_language_model(load_config_with_defaults(str(args.config), resolve=True))


if __name__ == "__main__":
    main()
