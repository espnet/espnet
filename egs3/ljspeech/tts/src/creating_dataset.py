"""Prepare the LJSpeech dataset for ESPnet3 TTS recipes."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from espnet2.text.build_tokenizer import build_tokenizer
from espnet3.utils.download_utils import download_url, setup_logger

LJSPEECH_URL = "http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def _ensure_downloaded(
    corpus_root: str | Path | None,
    dataset_dir: Path,
) -> Path:
    logger = setup_logger("ljspeech.tts.create_dataset", log_dir=dataset_dir)
    try:
        return _resolve_corpus_root(corpus_root)
    except FileNotFoundError:
        pass

    downloads_dir = dataset_dir / "downloads"
    archive_path = downloads_dir / "LJSpeech-1.1.tar.bz2"
    extracted_root = downloads_dir / "LJSpeech-1.1"

    if not extracted_root.is_dir():
        if not archive_path.is_file():
            download_url(LJSPEECH_URL, archive_path, logger=logger)
        logger.info("Extracting: %s", archive_path.name)
        downloads_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=downloads_dir)

    return _resolve_corpus_root(extracted_root)


def _resolve_corpus_root(corpus_root: str | Path | None) -> Path:
    candidates = []
    if corpus_root is not None:
        candidates.append(Path(corpus_root))
    env_root = os.environ.get("LJSPEECH")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            Path("downloads/LJSpeech-1.1"),
            Path("downloads") / "LJSpeech-1.1",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.name == "LJSpeech-1.1" and (candidate / "metadata.csv").is_file():
            return candidate
        if (candidate / "LJSpeech-1.1" / "metadata.csv").is_file():
            return candidate / "LJSpeech-1.1"

    raise FileNotFoundError(
        "LJSpeech corpus not found. Set LJSPEECH or create_dataset.corpus_root."
    )


def _write_manifest(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for utt_id, wav_path, text in rows:
            f.write(f"{utt_id}\t{wav_path}\t{text}\n")


def _build_token_list(
    texts: list[str],
    token_type: str,
    g2p_type: str | None,
) -> list[str]:
    tokenizer = build_tokenizer(token_type=token_type, g2p_type=g2p_type)
    tokens = sorted(
        {
            token
            for text in texts
            for token in tokenizer.text2tokens(text)
            if token != "<unk>"
        }
    )
    return ["<unk>", *tokens]


def create_dataset(
    dataset_dir: str | Path,
    corpus_root: str | Path | None = None,
    token_type: str = "phn",
    g2p_type: str | None = "g2p_en_no_space",
    token_list_path: str | Path | None = None,
) -> None:
    dataset_dir = Path(dataset_dir)
    ljspeech_root = _ensure_downloaded(corpus_root, dataset_dir)

    rows: list[tuple[str, str, str]] = []
    metadata = ljspeech_root / "metadata.csv"
    with metadata.open("r", encoding="utf-8") as f:
        for line in f:
            utt_id, _, normalized_text = line.rstrip("\n").split("|", maxsplit=2)
            wav_path = ljspeech_root / "wavs" / f"{utt_id}.wav"
            rows.append((utt_id, str(wav_path), normalized_text))

    if len(rows) <= 500:
        raise RuntimeError("LJSpeech split logic expects more than 500 utterances.")

    train_rows = rows[:-500]
    dev_rows = rows[-500:-250]
    eval_rows = rows[-250:]

    manifest_dir = dataset_dir / "manifest"
    _write_manifest(manifest_dir / "tr_no_dev.tsv", train_rows)
    _write_manifest(manifest_dir / "dev.tsv", dev_rows)
    _write_manifest(manifest_dir / "eval1.tsv", eval_rows)

    token_list = _build_token_list(
        [text for _, _, text in train_rows],
        token_type=token_type,
        g2p_type=g2p_type,
    )
    token_list_path = (
        Path(token_list_path)
        if token_list_path is not None
        else dataset_dir / "tokens.txt"
    )
    token_list_path.parent.mkdir(parents=True, exist_ok=True)
    token_list_path.write_text("\n".join(token_list) + "\n", encoding="utf-8")
