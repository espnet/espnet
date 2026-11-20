"""LibriCSS dataset builder.

Downloads and prepares the LibriCSS corpus and builds the recipe's ``data/``
directory:

.. code-block:: text

    data/wavs/<reco>.wav                          mono (one mic channel) audio
    data/<split>/wav.scp                          reco -> wav path
    data/<split>/oracle/segments                  utt reco start end
    data/<split>/oracle/utt2spk                   utt spk
    data/<split>/oracle/text                      utt transcript

where ``reco = <session>_CH<mic>_<condition>`` (e.g. ``session0_CH0_0L``),
``utt = <spk>_<reco>_<start_cs>_<end_cs>`` (centiseconds, zero-padded to 6
digits), split ``dev`` holds the configured dev sessions (session0 by
default) and split ``eval`` all remaining sessions.

Ported from the ESPnet1 recipe egs/libri_css/asr1 (PR espnet/espnet#2246):
``local/data_download.sh``, ``local/prepare_data.py`` and
``local/data_prep_mono.sh``.

LibriCSS: https://github.com/fgnt/LibriCSS
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

# Overlap conditions of the corpus (top-level directories under for_release/).
CONDITIONS = ("0L", "0S", "OV10", "OV20", "OV30", "OV40")

# LibriCSS ships ten ~10-minute sessions, each simulated under six recording
# conditions (~10 hours of audio in total): session0 is the conventional dev
# set, session1..session9 are eval.
ALL_SESSIONS = tuple(f"session{i}" for i in range(10))

# Google Drive file id of for_release.zip.
DEFAULT_GDRIVE_ID = "1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l"

# Environment variable pointing at an existing corpus directory (the one
# containing `for_release/`).
SOURCE_ENV_VAR = "LIBRICSS"

# Block size (frames) used when copying one microphone channel out of the
# 7-channel raw recordings without loading whole files into memory.
_COPY_BLOCKSIZE = 2_000_000


def _zfill6(seconds: float) -> str:
    """Format seconds as zero-padded centiseconds (egs1 convention)."""
    return "{:.0f}".format(100 * seconds).zfill(6)


def _round2(value: str | float) -> float:
    """Round a time value to 2 decimals, as egs1 prepare_data.py did."""
    return float("{:.2f}".format(float(value)))


def resolve_corpus_dir(
    recipe_dir: str | Path,
    source_dir: str | Path | None = None,
    download_dir: str | Path | None = None,
) -> Path:
    """Return the LibriCSS corpus directory (the one holding ``for_release``).

    Priority: explicit ``source_dir`` > ``$LIBRICSS`` > ``download_dir`` >
    ``<recipe_dir>/download/libricss``.
    """
    if source_dir is not None and str(source_dir):
        return Path(source_dir).resolve()
    env_path = os.environ.get(SOURCE_ENV_VAR)
    if env_path:
        return Path(env_path).resolve()
    if download_dir is not None and str(download_dir):
        return Path(download_dir).resolve()
    return (Path(recipe_dir) / "download" / "libricss").resolve()


def find_meetings(corpus_dir: Path) -> dict[str, list[Path]]:
    """Return ``{condition: [meeting dirs]}`` found under ``for_release``."""
    meetings: dict[str, list[Path]] = {}
    for cond in CONDITIONS:
        cond_dir = corpus_dir / "for_release" / cond
        found = sorted(
            meet
            for meet in cond_dir.glob("overlap*")
            if (meet / "record" / "raw_recording.wav").is_file()
            and (meet / "transcription" / "meeting_info.txt").is_file()
        )
        meetings[cond] = found
    return meetings


def is_corpus_prepared(corpus_dir: Path) -> bool:
    """Check that all six conditions expose their per-session meetings."""
    if not (corpus_dir / "for_release").is_dir():
        return False
    meetings = find_meetings(corpus_dir)
    n_sessions = len(ALL_SESSIONS)
    return all(len(meetings[cond]) >= n_sessions for cond in CONDITIONS)


class LibriCSSBuilder(DatasetBuilder):
    """Download/prepare LibriCSS and build the recipe ``data/`` directory.

    All methods accept the keys of the ``create_dataset`` config block plus
    arbitrary extra keys (absorbed by ``**_kwargs``), so template defaults
    such as ``func``/``dataset_dir`` pass through harmlessly.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        download_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether the raw corpus tree is ready under ``for_release``."""
        corpus_dir = resolve_corpus_dir(recipe_dir, source_dir, download_dir)
        return is_corpus_prepared(corpus_dir)

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        download_dir: str | Path | None = None,
        gdrive_id: str = DEFAULT_GDRIVE_ID,
        **_kwargs,
    ) -> None:
        """Download for_release.zip and unzip it.

        Port of egs1 ``local/data_download.sh`` (wget-with-cookies replaced by
        ``gdown``). egs1 also ran the corpus-shipped ``segment_libricss.py``
        after unzipping; this recipe skips it because that script only adds
        ``utterances/``/``segments/`` wavs (~15 GiB) which the ASR flow never
        reads -- the release zip already contains the two inputs the builder
        needs (``record/raw_recording.wav`` and
        ``transcription/meeting_info.txt``). When an explicit ``source_dir`` /
        ``$LIBRICSS`` is given and incomplete, this raises instead of
        downloading into a user-provided location.
        """
        corpus_dir = resolve_corpus_dir(recipe_dir, source_dir, download_dir)
        if is_corpus_prepared(corpus_dir):
            logger.info("LibriCSS corpus already prepared: %s", corpus_dir)
            return

        explicit_source = bool(source_dir and str(source_dir)) or bool(
            os.environ.get(SOURCE_ENV_VAR)
        )
        if explicit_source:
            raise FileNotFoundError(
                f"LibriCSS source at {corpus_dir} is incomplete: expected "
                f"for_release/{{{','.join(CONDITIONS)}}}/overlap*/"
                "{record/raw_recording.wav,transcription/meeting_info.txt}. "
                f"Unset {SOURCE_ENV_VAR}/create_dataset.source_dir to let "
                "this recipe download the corpus instead."
            )

        corpus_dir.mkdir(parents=True, exist_ok=True)
        zip_path = corpus_dir / "for_release.zip"

        if not zip_path.is_file():
            try:
                import gdown
            except ImportError as e:
                raise RuntimeError(
                    "Downloading LibriCSS requires gdown: `pip install gdown`. "
                    f"Alternatively, place the corpus under {corpus_dir} or "
                    f"set the {SOURCE_ENV_VAR} environment variable."
                ) from e
            logger.info(
                "Downloading LibriCSS for_release.zip from Google Drive "
                "(id=%s) to %s",
                gdrive_id,
                zip_path,
            )
            gdown.download(id=str(gdrive_id), output=str(zip_path))

        for_release = corpus_dir / "for_release"
        if not for_release.is_dir():
            logger.info("Unzipping %s", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(corpus_dir)

        if not is_corpus_prepared(corpus_dir):
            raise RuntimeError(
                f"LibriCSS preparation failed: {corpus_dir}/for_release does "
                f"not contain all {len(CONDITIONS)} conditions x "
                f"{len(ALL_SESSIONS)} sessions with record/raw_recording.wav "
                f"+ transcription/meeting_info.txt."
            )
        logger.info("LibriCSS corpus prepared: %s", corpus_dir)

    def is_built(
        self,
        recipe_dir: str | Path,
        dev_sessions: list[str] | tuple[str, ...] = ("session0",),
        **_kwargs,
    ) -> bool:
        """Check that ``data/`` holds wav.scp + oracle files for both splits."""
        data_dir = Path(recipe_dir) / "data"
        dev_sessions = {str(s) for s in dev_sessions}
        expected = {
            "dev": len(CONDITIONS) * len(dev_sessions),
            "eval": len(CONDITIONS) * (len(ALL_SESSIONS) - len(dev_sessions)),
        }
        for split, n_recos in expected.items():
            wav_scp = data_dir / split / "wav.scp"
            if not wav_scp.is_file():
                return False
            for name in ("segments", "utt2spk", "text"):
                if not (data_dir / split / "oracle" / name).is_file():
                    return False
            n_lines = sum(1 for line in wav_scp.open() if line.strip())
            if n_lines != n_recos:
                return False
        return True

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        download_dir: str | Path | None = None,
        mic: int = 0,
        dev_sessions: list[str] | tuple[str, ...] = ("session0",),
        **_kwargs,
    ) -> None:
        """Extract one mic channel per meeting and write oracle data files.

        Port of egs1 ``local/prepare_data.py`` (channel extraction and
        meeting_info.txt parsing) and ``local/data_prep_mono.sh`` (dev/eval
        session split).
        """
        recipe_dir = Path(recipe_dir)
        corpus_dir = resolve_corpus_dir(recipe_dir, source_dir, download_dir)
        if not is_corpus_prepared(corpus_dir):
            raise FileNotFoundError(
                f"LibriCSS corpus not prepared at {corpus_dir}; run the "
                "create_dataset stage (prepare_source) first."
            )
        mic = int(mic)
        dev_sessions = {str(s) for s in dev_sessions}
        unknown = dev_sessions - set(ALL_SESSIONS)
        if unknown:
            raise ValueError(
                f"Unknown dev sessions {sorted(unknown)}; LibriCSS provides "
                f"{list(ALL_SESSIONS)}."
            )

        data_dir = recipe_dir / "data"
        wav_dir = data_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        for split in ("dev", "eval"):
            (data_dir / split / "oracle").mkdir(parents=True, exist_ok=True)

        # reco -> (split, wav path); utt-level rows collected per split.
        rows: dict[str, dict[str, list[str]]] = {
            "dev": {"wav.scp": [], "segments": [], "utt2spk": [], "text": []},
            "eval": {"wav.scp": [], "segments": [], "utt2spk": [], "text": []},
        }

        meetings = find_meetings(corpus_dir)
        n_total = sum(len(v) for v in meetings.values())
        n_done = 0
        for cond in CONDITIONS:
            for meet in meetings[cond]:
                n_done += 1
                # Meeting dirs are named
                # overlap_ratio_<rate>_sil2.9_3.0_<session>_actual0.0,
                # i.e. egs1's `_, _, _, _, _, sessid, olr = name.split("_")`
                # (7 parts; sessid = parts[-2]).
                parts = meet.name.split("_")
                if len(parts) < 7:
                    raise ValueError(f"Unexpected meeting directory name: {meet}")
                sessid = parts[-2]
                split = "dev" if sessid in dev_sessions else "eval"
                reco_id = f"{sessid}_CH{mic}_{cond}"

                raw_wav = meet / "record" / "raw_recording.wav"
                out_wav = (wav_dir / f"{reco_id}.wav").resolve()
                logger.info(
                    "[%d/%d] %s: extracting mic %d -> %s",
                    n_done,
                    n_total,
                    meet.name,
                    mic,
                    out_wav.name,
                )
                self._extract_channel(raw_wav, out_wav, mic)
                rows[split]["wav.scp"].append(f"{reco_id} {out_wav}")

                info_path = meet / "transcription" / "meeting_info.txt"
                for utt_id, spkid, start, end, text in self._parse_meeting_info(
                    info_path, reco_id
                ):
                    rows[split]["segments"].append(
                        f"{utt_id} {reco_id} {start:.2f} {end:.2f}"
                    )
                    rows[split]["utt2spk"].append(f"{utt_id} {spkid}")
                    rows[split]["text"].append(f"{utt_id} {text}")

        for split, files in rows.items():
            for name, lines in files.items():
                path = (
                    data_dir / split / name
                    if name == "wav.scp"
                    else data_dir / split / "oracle" / name
                )
                lines = sorted(set(lines))
                with path.open("w", encoding="utf-8") as f:
                    f.write("".join(f"{line}\n" for line in lines))
                logger.info("Wrote %s (%d lines)", path, len(lines))

    @staticmethod
    def _extract_channel(raw_wav: Path, out_wav: Path, mic: int) -> None:
        """Copy one channel of a multi-channel wav to a mono PCM_16 file."""
        info = sf.info(str(raw_wav))
        if mic >= info.channels:
            raise ValueError(
                f"{raw_wav} has {info.channels} channels; mic={mic} requested."
            )
        with (
            sf.SoundFile(str(raw_wav)) as fin,
            sf.SoundFile(
                str(out_wav),
                mode="w",
                samplerate=info.samplerate,
                channels=1,
                subtype="PCM_16",
            ) as fout,
        ):
            for block in fin.blocks(blocksize=_COPY_BLOCKSIZE, dtype="float32"):
                fout.write(block[:, mic])

    @staticmethod
    def _parse_meeting_info(info_path: Path, reco_id: str):
        """Yield ``(utt_id, spkid, start, end, text)`` from meeting_info.txt.

        The file is tab-separated; after one header line each row is
        ``start end spkid uttid text`` (``uttid`` is the original LibriSpeech
        utterance id; egs1 dropped it and so do we).
        """
        with info_path.open("r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=4)
                if len(parts) < 4:
                    raise ValueError(f"Malformed line in {info_path}: {line!r}")
                start = _round2(parts[0])
                end = _round2(parts[1])
                spkid = parts[2]
                text = parts[4] if len(parts) == 5 else ""
                utt_id = f"{spkid}_{reco_id}_{_zfill6(start)}_{_zfill6(end)}"
                yield utt_id, spkid, start, end, text
