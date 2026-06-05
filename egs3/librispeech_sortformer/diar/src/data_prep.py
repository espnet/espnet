"""Data preparation for the 8-speaker streaming Sortformer recipe.

Stage 1 of the recipe. Two parts:

1. **FastMSS simulated meetings** (the main training data): generate 1-minute,
   3-8-speaker simulated LibriSpeech conversations (reverberant, with additive
   noise) by shelling out to FastMSS ``sim_librispeech.py`` with Hydra overrides.
   Produces a lhotse CutSet with multi-speaker supervisions.

2. **AMI single-distant-mic (Array1-01) cuts**: 1-minute windows of AMI SDM for
   additional training data, plus dev/test for long-form evaluation, built from
   the AMI lhotse recordings + supervisions.

Invoked by ``DiarizationSystem.data_preparation`` via the ``data_prep`` block in
``conf/training.yaml``.
"""

import logging
import subprocess
import zipfile
from pathlib import Path

import lhotse

logger = logging.getLogger(__name__)

# CorentinJ / lhotse LibriSpeech word alignments (.txt format expected by
# lhotse.recipes.prepare_librispeech; NOT the MFA .TextGrid format).
LIBRISPEECH_ALIGNMENTS_URL = (
    "https://drive.google.com/uc?id=1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE"
)


def download_librispeech_alignments(target_dir):
    """Download + extract the lhotse-format LibriSpeech alignments. Returns the
    ``.../LibriSpeech-Alignments/LibriSpeech`` dir to pass as ``librispeech_align``.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = target_dir / "LibriSpeech-Alignments" / "LibriSpeech"
    if extracted.is_dir():
        return str(extracted)
    import gdown

    zp = target_dir / "LibriSpeech-Alignments.zip"
    if not zp.is_file():
        logger.info("Downloading LibriSpeech alignments -> %s", zp)
        try:
            gdown.download(
                LIBRISPEECH_ALIGNMENTS_URL, output=str(zp), quiet=False, fuzzy=True
            )
        except Exception as e:
            raise RuntimeError(
                "Could not download LibriSpeech alignments from Google Drive "
                f"({e}). The public link is frequently quota-limited. Either retry "
                "later, or download 'LibriSpeech-Alignments.zip' manually (lhotse "
                "LIBRISPEECH_ALIGNMENTS_URL / github.com/CorentinJ/librispeech-"
                "alignments), unzip it, and set data_prep.fastmss.librispeech_align "
                "to the extracted '.../LibriSpeech-Alignments/LibriSpeech' dir."
            ) from e
    with zipfile.ZipFile(zp) as f:
        f.extractall(target_dir)
    assert extracted.is_dir(), f"Unexpected alignments layout under {target_dir}"
    return str(extracted)


def build_ami_cuts(
    ami_dir, data_dir, cond="sdm", window=60.0, splits=("train", "dev", "test")
):
    """Build windowed AMI cuts (with supervisions) for a mic condition."""
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
    extra_overrides=None,
):
    """Generate FastMSS LibriSpeech meetings via Hydra overrides on sim_librispeech.

    Returns the path to the generated synthetic CutSet.
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

    def _list(x):
        return "[" + ",".join(str(v) for v in x) + "]"

    overrides = [
        "stage=0",
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
    ]
    if noise_folders:
        overrides += ["add_noise=true", f"noise_folders={_list(noise_folders)}"]
    else:
        overrides += ["add_noise=false", "noise_folders=null"]
    overrides += list(extra_overrides or [])

    cmd = ["python", str(script), "--config-name", "librispeech"] + overrides
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
    """Top-level data-prep entrypoint (called by DiarizationSystem)."""
    build_ami_cuts(ami_dir, data_dir, cond=ami_cond, window=window, splits=ami_splits)
    if fastmss is not None:
        # Resolve the LibriSpeech alignments: auto-download the lhotse-format
        # (.txt) alignments if the configured path is missing or set to "auto".
        align = fastmss.get("librispeech_align")
        if align in (None, "auto") or not Path(align).is_dir():
            logger.info("librispeech_align missing/auto -> downloading alignments.")
            align = download_librispeech_alignments(Path(data_dir) / "ls_alignments")
        synth = run_fastmss(
            fastmss_dir=fastmss["dir"],
            output_dir=fastmss.get("output_dir", str(Path(data_dir) / "fastmss")),
            librispeech_dir=fastmss["librispeech_dir"],
            librispeech_align=align,
            noise_folders=fastmss.get("noise_folders"),
            n_meetings=fastmss.get("n_meetings", 2000),
            min_max_spk=fastmss.get("min_max_spk", (3, 8)),
            duration=fastmss.get("duration", int(window)),
            dset_splits=fastmss.get("dset_splits", ("train-clean-100",)),
            n_jobs=fastmss.get("n_jobs", 16),
            extra_overrides=fastmss.get("overrides"),
        )
        logger.info(
            "Point dataset/config.yaml splits.train at the FastMSS cuts: %s "
            "(combined with the AMI train cuts via the DataOrganizer train list).",
            synth,
        )
