"""Data preparation for the 8-speaker streaming Sortformer recipe.

Stage 1 of the recipe. Two parts:

1. **FastMSS simulated meetings** (the main training data): generate 90 s,
   3-8-speaker simulated LibriSpeech conversations (reverberant, with additive
   noise) by shelling out to FastMSS ``sim_librispeech.py`` with Hydra overrides.
   Produces a lhotse CutSet with multi-speaker supervisions.

2. **AMI single-distant-mic (Array1-01) cuts**: 90 s windows of AMI SDM for
   additional training data, plus dev/test for long-form evaluation, built from
   the AMI lhotse recordings + supervisions.

The stage is wired through the conf resolver as
``data_prep.func: src.data_prep.prepare``: ``DiarizationSystem.data_preparation``
resolves that dotted path and calls :func:`prepare` with the ``data_prep.*``
keyword arguments from ``conf/training.yaml``.
"""

import logging
import subprocess
import sys
from pathlib import Path

import lhotse

logger = logging.getLogger(__name__)


def download_librispeech_alignments(target_dir):
    """Fetch the lhotse-format LibriSpeech word alignments (``.txt``).

    Delegates to ``lhotse.recipes.librispeech.download_librispeech`` with an empty
    ``dataset_parts`` so it downloads *only* the alignments (not the 12 GB audio).
    If the alignments already exist under ``target_dir`` the download is skipped.

    Args:
        target_dir: Directory to download/extract the alignments into.

    Returns:
        The ``.../LibriSpeech-Alignments/LibriSpeech`` directory, suitable to
        pass to FastMSS as ``librispeech_align``.

    Raises:
        RuntimeError: If the download fails (the upstream Google-Drive link is
            frequently quota-limited; the message explains the manual fallback).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = target_dir / "LibriSpeech-Alignments" / "LibriSpeech"
    if extracted.is_dir():
        return str(extracted)
    from lhotse.recipes.librispeech import download_librispeech

    try:
        download_librispeech(target_dir=target_dir, dataset_parts=[], alignments=True)
    except Exception as e:
        raise RuntimeError(
            "Could not download LibriSpeech alignments via lhotse "
            f"(download_librispeech, alignments=True): {e}. The Google-Drive link "
            "is frequently quota-limited -- retry later, or download "
            "'LibriSpeech-Alignments.zip' manually, unzip it, and set "
            "data_prep.fastmss.librispeech_align to the extracted "
            "'.../LibriSpeech-Alignments/LibriSpeech' dir."
        ) from e
    assert extracted.is_dir(), f"Unexpected alignments layout under {target_dir}"
    return str(extracted)


def build_ami_cuts(
    ami_dir, data_dir, cond="sdm", window=60.0, splits=("train", "dev", "test")
):
    """Build windowed AMI cuts (with supervisions) for a mic condition.

    For each split, loads the AMI lhotse recording/supervision manifests, forms a
    ``CutSet``, slices it into fixed ``window``-second windows and keeps only
    windows that contain at least one supervision. Splits whose manifests are
    missing are skipped with a warning. Each split's cuts are written to
    ``<data_dir>/ami_<cond>_<split>_<window>s_cuts.jsonl.gz``.

    Args:
        ami_dir: Directory holding ``ami-<cond>_{recordings,supervisions}_<split>``
            ``.jsonl.gz`` manifests.
        data_dir: Output directory for the windowed cut files (created if needed).
        cond: AMI mic condition, e.g. ``"sdm"`` (single distant mic).
        window: Window length in seconds.
        splits: Splits to build (subset of ``train``/``dev``/``test``).

    Returns:
        A dict mapping each built split to its output cut-file path.
    """
    ami_dir = Path(ami_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in splits:
        rec_p = ami_dir / f"ami-{cond}_recordings_{split}.jsonl.gz"
        sup_p = ami_dir / f"ami-{cond}_supervisions_{split}.jsonl.gz"
        if not rec_p.is_file() or not sup_p.is_file():
            logger.warning("Missing AMI %s %s manifests; skipping.", cond, split)
            continue
        recs = lhotse.load_manifest(str(rec_p))
        sups = lhotse.load_manifest(str(sup_p))
        cuts = lhotse.CutSet.from_manifests(recordings=recs, supervisions=sups)
        cuts = cuts.cut_into_windows(duration=window)
        cuts = cuts.filter(lambda c: len(c.supervisions) > 0).to_eager()
        out_path = data_dir / f"ami_{cond}_{split}_{int(window)}s_cuts.jsonl.gz"
        cuts.to_file(str(out_path))
        logger.info(
            "Wrote %d AMI %s %s windows -> %s", len(cuts), cond, split, out_path
        )
        out[split] = str(out_path)
    return out


def build_source_cuts(aligned_manifests, output_dir, dset_splits, prefix="librispeech"):
    """Pre-build FastMSS's ``all_cuts_orig.jsonl.gz`` from word-aligned manifests.

    Combines the per-split word-aligned lhotse recording/supervision manifests
    into a single ``CutSet`` and writes it where FastMSS expects its source cuts,
    so FastMSS can skip stage 0 (alignment download / preparation).

    Args:
        aligned_manifests: Directory of word-aligned lhotse manifests named
            ``<prefix>_{recordings,supervisions}_<split>.jsonl.gz``.
        output_dir: FastMSS output dir; the cuts are written under its
            ``manifests/`` subdirectory.
        dset_splits: Splits to combine (e.g. ``("train-clean-100",)``).
        prefix: Manifest filename prefix (default ``"librispeech"``).

    Returns:
        The combined source ``CutSet``.
    """
    from lhotse import CutSet
    from lhotse.manipulation import combine as combine_manifests

    aligned = Path(aligned_manifests)
    parts = []
    for split in dset_splits:
        rec = lhotse.load_manifest(
            str(aligned / f"{prefix}_recordings_{split}.jsonl.gz")
        )
        sup = lhotse.load_manifest(
            str(aligned / f"{prefix}_supervisions_{split}.jsonl.gz")
        )
        parts.append(CutSet.from_manifests(recordings=rec, supervisions=sup))
    all_cuts = combine_manifests(parts)
    mdir = Path(output_dir) / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    all_cuts.to_file(str(mdir / "all_cuts_orig.jsonl.gz"))
    logger.info(
        "Pre-built source cuts (%d) from aligned manifests -> %s",
        len(all_cuts),
        mdir / "all_cuts_orig.jsonl.gz",
    )
    return all_cuts


def run_fastmss(
    fastmss_dir,
    output_dir,
    librispeech_dir,
    librispeech_align,
    noise_folders=None,
    n_meetings=2000,
    min_max_spk=(3, 8),
    duration=60,
    dset_splits=("train-clean-100",),
    samplerate=16000,
    reverberate=True,
    n_jobs=16,
    aligned_manifests=None,
    extra_overrides=None,
):
    """Generate FastMSS LibriSpeech meetings via Hydra overrides on sim_librispeech.

    Shells out to FastMSS ``sim_librispeech.py`` (config ``librispeech``) with the
    given parameters passed as Hydra overrides, producing reverberant,
    optionally noisy multi-speaker LibriSpeech meetings as a lhotse ``CutSet``.
    Only the mixture and supervisions are saved (anechoic / per-speaker source
    streams are disabled, as diarization does not need them).

    Source-alignment handling has two modes:
        * ``aligned_manifests`` given -- pre-build the source cuts with
          :func:`build_source_cuts` and start FastMSS at stage 1, skipping the
          alignment download/prep entirely. ``librispeech_align`` is then unused.
        * ``aligned_manifests`` is ``None`` -- start at stage 0 and let FastMSS
          prepare LibriSpeech using the ``librispeech_align`` alignments
          directory (see :func:`download_librispeech_alignments` and the
          ``librispeech_align: auto`` option handled in :func:`prepare`).

    Args:
        fastmss_dir: Path to a FastMSS checkout containing ``sim_librispeech.py``.
        output_dir: FastMSS output directory (created if needed).
        librispeech_dir: Path to the LibriSpeech audio root.
        librispeech_align: Directory of lhotse-format word alignments (ignored
            when ``aligned_manifests`` is given).
        noise_folders: Optional list of noise directories; if set, additive noise
            is enabled.
        n_meetings: Number of meetings to simulate.
        min_max_spk: ``(min, max)`` speakers per meeting.
        duration: Meeting duration in seconds.
        dset_splits: LibriSpeech splits to draw sources from.
        samplerate: Output sample rate in Hz.
        reverberate: Whether to apply simulated room reverberation.
        n_jobs: Parallel worker count for FastMSS.
        aligned_manifests: Optional dir of word-aligned lhotse manifests; see the
            two modes above.
        extra_overrides: Optional list of extra raw Hydra override strings.

    Returns:
        Path to the generated synthetic train ``CutSet``
        (``<output_dir>/manifests/synth-librispeech-train-cuts.jsonl.gz``).

    Raises:
        FileNotFoundError: If ``sim_librispeech.py`` is not found under
            ``fastmss_dir``.
    """
    fastmss_dir = Path(fastmss_dir)
    script = fastmss_dir / "sim_librispeech.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"FastMSS script not found: {script}. Clone "
            "https://github.com/popcornell/FastMSS (branch `librispeech`) and set "
            "data_prep.fastmss.dir to that checkout."
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_stage = 0
    if aligned_manifests is not None:
        build_source_cuts(aligned_manifests, output_dir, dset_splits)
        start_stage = 1  # skip prepare_librispeech / alignment download

    def _list(x):
        return "[" + ",".join(str(v) for v in x) + "]"

    overrides = [
        f"stage={start_stage}",
        f"samplerate={samplerate}",
        f"output_dir={output_dir}",
        f"n_meetings={n_meetings}",
        f"min_max_spk={_list(min_max_spk)}",
        f"duration={duration}",
        f"librispeech_dir={librispeech_dir}",
        f"librispeech_align={librispeech_align}",
        f"dset_splits={_list(dset_splits)}",
        f"reverberate={str(reverberate).lower()}",
        f"n_jobs={n_jobs}",
        "save_cutset=true",
        # Diarization only needs the mixture + supervisions, not the anechoic /
        # per-speaker source streams (those are for separation).
        "save_anechoic=false",
        "save_spk=false",
    ]
    if noise_folders:
        overrides += ["add_noise=true", f"noise_folders={_list(noise_folders)}"]
    else:
        overrides += ["add_noise=false", "noise_folders=null"]
    overrides += list(extra_overrides or [])

    cmd = [sys.executable, str(script), "--config-name", "librispeech"] + overrides
    logger.info("Running FastMSS: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(fastmss_dir), check=True)

    synth = Path(output_dir) / "manifests" / "synth-librispeech-train-cuts.jsonl.gz"
    logger.info("FastMSS synthetic cuts: %s", synth)
    return str(synth)


def prepare(
    data_dir,
    ami_dir,
    ami_cond="sdm",
    window=60.0,
    ami_splits=("train", "dev", "test"),
    fastmss=None,
):
    """Top-level data-prep entrypoint for the recipe's stage 1.

    Resolved from ``conf/training.yaml`` as ``data_prep.func:
    src.data_prep.prepare`` and called with the remaining ``data_prep.*`` keys.
    It always builds the AMI cuts, and additionally generates FastMSS meetings
    when a ``fastmss`` block is provided.

    Args:
        data_dir: Output directory for the recipe's prepared data.
        ami_dir: Directory of AMI lhotse manifests (see :func:`build_ami_cuts`).
        ami_cond: AMI mic condition, e.g. ``"sdm"``.
        window: Window length in seconds for AMI (and the FastMSS default
            duration).
        ami_splits: AMI splits to build.
        fastmss: Optional dict configuring the FastMSS stage. Recognized keys:
            ``dir``, ``librispeech_dir`` (required when present), ``output_dir``,
            ``noise_folders``, ``n_meetings``, ``min_max_spk``, ``duration``,
            ``dset_splits``, ``n_jobs``, ``overrides``, and the source-alignment
            keys ``aligned_manifests`` and ``librispeech_align``. Prefer
            ``aligned_manifests`` (skips the alignment download). Otherwise
            ``librispeech_align`` is resolved, auto-downloading the lhotse-format
            (``.txt``) alignments when missing or set to ``"auto"``.

    Note:
        FastMSS train cuts are written but not registered automatically; point
        ``dataset/config.yaml`` ``splits.train`` at them (combined with the AMI
        train cuts via the DataOrganizer train list).
    """
    build_ami_cuts(ami_dir, data_dir, cond=ami_cond, window=window, splits=ami_splits)
    if fastmss is not None:
        # Prefer pre-aligned lhotse manifests (skips alignment download). If not
        # given, resolve librispeech_align, auto-downloading the lhotse-format
        # (.txt) alignments when missing or set to "auto".
        aligned_manifests = fastmss.get("aligned_manifests")
        align = fastmss.get("librispeech_align")
        if aligned_manifests is None:
            if align in (None, "auto") or not Path(align).is_dir():
                logger.info("No aligned_manifests; downloading LS alignments.")
                align = download_librispeech_alignments(
                    Path(data_dir) / "ls_alignments"
                )
        synth = run_fastmss(
            fastmss_dir=fastmss["dir"],
            output_dir=fastmss.get("output_dir", str(Path(data_dir) / "fastmss")),
            librispeech_dir=fastmss["librispeech_dir"],
            librispeech_align=align or "unused",
            noise_folders=fastmss.get("noise_folders"),
            n_meetings=fastmss.get("n_meetings", 2000),
            min_max_spk=fastmss.get("min_max_spk", (3, 8)),
            duration=fastmss.get("duration", int(window)),
            dset_splits=fastmss.get("dset_splits", ("train-clean-100",)),
            n_jobs=fastmss.get("n_jobs", 16),
            aligned_manifests=aligned_manifests,
            extra_overrides=fastmss.get("overrides"),
        )
        logger.info(
            "Point dataset/config.yaml splits.train at the FastMSS cuts: %s "
            "(combined with the AMI train cuts via the DataOrganizer train list).",
            synth,
        )
