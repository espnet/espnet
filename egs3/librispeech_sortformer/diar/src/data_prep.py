"""Data preparation for the Sortformer diarization recipe.

Two responsibilities:

1. **AMI evaluation cuts** (always): build fixed-length (default 30 s) windows of
   the AMI *mixed-headset* recordings, each carrying its overlapping speaker
   supervisions, from the lhotse manifests in ``ami_dir``. The pre-built
   ``ami-ihm-mix_cutset_*_30s`` manifests ship *without* supervisions, so we
   reconstruct them here from ``ami-ihm-mix_recordings_*`` +
   ``ami-ihm-mix_supervisions_*``.

2. **FastMSS training mixtures** (optional): if ``fastmss`` is configured, shell
   out to FastMSS ``sim_librispeech.py`` to generate simulated LibriSpeech
   meeting mixtures and expose them as the ``train`` CutSet. Requires a FastMSS
   checkout plus LibriSpeech/alignments/noise on disk (see FastMSS README).

Invoked by ``DiarizationSystem.data_preparation`` via the ``data_prep`` block in
``conf/training.yaml``.
"""

import logging
import subprocess
from pathlib import Path

import lhotse

logger = logging.getLogger(__name__)


def build_ami_eval_cuts(
    ami_dir,
    data_dir,
    window=30.0,
    splits=("dev", "test"),
    manifest_prefix="ami-ihm-mix",
):
    """Build AMI mixed-headset windowed cuts with supervisions."""
    ami_dir = Path(ami_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in splits:
        rec_p = ami_dir / f"{manifest_prefix}_recordings_{split}.jsonl.gz"
        sup_p = ami_dir / f"{manifest_prefix}_supervisions_{split}.jsonl.gz"
        if not rec_p.is_file() or not sup_p.is_file():
            logger.warning(
                "Missing AMI manifests for split %s (%s / %s); skipping.",
                split,
                rec_p,
                sup_p,
            )
            continue
        recs = lhotse.load_manifest(str(rec_p))
        sups = lhotse.load_manifest(str(sup_p))
        cuts = lhotse.CutSet.from_manifests(recordings=recs, supervisions=sups)
        cuts = cuts.cut_into_windows(duration=window)
        # Keep windows that contain at least one supervision.
        cuts = cuts.filter(lambda c: len(c.supervisions) > 0).to_eager()
        name = {"dev": "valid", "test": "test"}.get(split, split)
        out_path = data_dir / f"ami_{split}_{int(window)}s_cuts.jsonl.gz"
        cuts.to_file(str(out_path))
        logger.info("Wrote %d AMI %s windows -> %s", len(cuts), split, out_path)
        out[name] = str(out_path)
    return out


def run_fastmss(fastmss_dir, config_name, overrides=None):
    """Run FastMSS sim_librispeech.py to simulate training mixtures."""
    fastmss_dir = Path(fastmss_dir)
    script = fastmss_dir / "sim_librispeech.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"FastMSS script not found: {script}. Clone "
            "https://github.com/popcornell/FastMSS (branch `librispeech`) and "
            "set data_prep.fastmss.dir."
        )
    cmd = ["python", str(script), "--config-name", config_name]
    cmd += list(overrides or [])
    logger.info("Running FastMSS: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(fastmss_dir), check=True)


def prepare(
    data_dir,
    ami_dir,
    window=30.0,
    ami_splits=("dev", "test"),
    fastmss=None,
):
    """Top-level data-prep entrypoint (called by DiarizationSystem)."""
    build_ami_eval_cuts(ami_dir, data_dir, window=window, splits=ami_splits)
    if fastmss is not None:
        run_fastmss(
            fastmss_dir=fastmss["dir"],
            config_name=fastmss.get("config_name", "librispeech"),
            overrides=fastmss.get("overrides"),
        )
        logger.info(
            "FastMSS finished. Point dataset/config.yaml `splits.train` at the "
            "generated synth cuts (e.g. <output_dir>/manifests/"
            "synth-librispeech-train-cuts.jsonl.gz)."
        )
