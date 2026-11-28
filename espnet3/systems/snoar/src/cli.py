"""``espnet snoar``: adapt a speech enhancer to your own noisy recordings.

    espnet snoar adapt recordings/ -o my_enhancer/
    espnet snoar enhance noisy.wav -o clean.wav --model my_enhancer/

``adapt`` needs nothing but the noisy audio. It starts from a seed
(``--seed``: an ESPnet2 enhancement tag, by default the URGENT 2025
TF-GridNet, or a directory exported by this system) and self-trains it with
SNOAR on the recordings. With ``--seed dnf`` or ``--seed mixit`` the seed is
itself trained first, label-free, from the recordings plus noise-only audio:
an external noise corpus (``--noise``) or the speech-free gaps of the
recordings found from VAD labels (``--vad file.rttm``) or an energy VAD.

The other verbs expose the steps on their own: ``seed`` trains a DNF or MixIT
seed, ``harvest`` writes a noise pool from VAD gaps, ``manifest`` lists
recordings. Every training verb writes its config to ``<out>/work/``, where
it can be rerun or edited like a recipe config.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

NOISY = "espnet3.systems.snoar.src.data.noisy"


class SNOARCLIError(RuntimeError):
    """A problem the user can fix, reported without a traceback."""


def load_cli_defaults():
    """Return the CLI defaults (``egs3/TEMPLATE/snoar/conf/cli.yaml``)."""
    res = resources.files("egs3.TEMPLATE.snoar").joinpath("conf/cli.yaml")
    with resources.as_file(res) as path:
        return OmegaConf.load(path)


# --------------------------------------------------------------------------- #
# Manifests and noise pools
# --------------------------------------------------------------------------- #
def write_recording_manifests(
    audio: str, work: Path, fs: int, valid_fraction: float, valid_max: int
) -> Dict[str, Path]:
    """List ``audio``, check its rate, and write train/valid manifests.

    Raises:
        SNOARCLIError: If there is no audio, or a file has another rate.
    """
    from espnet3.systems.snoar.src.data.manifest import (
        describe_recordings,
        list_audio,
        write_manifest,
    )

    try:
        rows = describe_recordings(list_audio(audio))
    except FileNotFoundError as e:
        raise SNOARCLIError(str(e)) from e
    bad = [r["audio"] for r in rows if r["sample_rate"] != fs]
    if bad:
        raise SNOARCLIError(
            f"{len(bad)} file(s) are not {fs} Hz (e.g. {bad[0]}); resample them "
            "or pass --fs"
        )
    _make_ids_unique(rows)
    n_valid = min(valid_max, max(1, int(len(rows) * valid_fraction)))
    if len(rows) < 2:
        n_valid = 0
    ranked = sorted(rows, key=lambda r: hashlib.sha1(r["audio"].encode()).hexdigest())
    valid, train = ranked[:n_valid], ranked[n_valid:]
    paths = {"train": work / "manifest" / "train.jsonl"}
    write_manifest(train, paths["train"])
    # A single recording is used for both; validation is monitoring only.
    paths["valid"] = work / "manifest" / "valid.jsonl"
    write_manifest(valid or train, paths["valid"])
    logger.info("recordings: %d train, %d valid", len(train), len(valid))
    return paths


def _make_ids_unique(rows: List[dict]) -> None:
    seen: Dict[str, int] = {}
    for r in rows:
        n = seen.get(r["id"], 0)
        seen[r["id"]] = n + 1
        if n:
            r["id"] = f"{r['id']}_{n}"


def write_noise_pool(
    noise: Optional[str],
    vad: Optional[str],
    train_manifest: Path,
    work: Path,
    fs: int,
    harvest_cfg,
) -> tuple[Path, str]:
    """Write the noise pool for a DNF/MixIT seed; return (path, noise_level).

    ``--noise`` (a directory or list of noise-only files) wins over ``--vad``;
    a ``.jsonl`` given to ``--noise`` is used as a ready pool. Without either,
    the energy VAD harvests the recordings' own gaps.

    Raises:
        SNOARCLIError: If no noise-only audio is found.
    """
    from espnet3.systems.snoar.src.data.manifest import (
        list_audio,
        read_manifest,
        write_manifest,
    )
    from espnet3.systems.snoar.src.data.vad import harvest_noise

    out = work / "manifest" / "noise.jsonl"
    if noise and noise.endswith(".jsonl"):
        return Path(noise), "native"
    if noise:
        try:
            files = list_audio(noise)
        except FileNotFoundError as e:
            raise SNOARCLIError(str(e)) from e
        rows = [
            {"audio": str(Path(f).resolve()), "start": 0, "end": None} for f in files
        ]
        write_manifest(rows, out)
        return out, "snr"
    rows = harvest_noise(
        read_manifest(train_manifest),
        vad=vad or "energy",
        fs=fs,
        margin=float(harvest_cfg.margin),
        min_gap=float(harvest_cfg.min_gap),
    )
    if not rows:
        raise SNOARCLIError(
            "the VAD found no speech-free gap of at least "
            f"{harvest_cfg.min_gap} s; pass --noise <noise-only audio>"
        )
    write_manifest(rows, out)
    hours = sum(r["end"] - r["start"] for r in rows) / fs / 3600
    print(
        f"[snoar] noise pool: {len(rows)} gaps, {hours:.2f} h ({vad or 'energy'} VAD)",
        file=sys.stderr,
    )
    return out, "native"


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #
def _trainer_overrides(device: str, steps: int) -> dict:
    import torch

    gpu = device != "cpu" and torch.cuda.is_available()
    if device.startswith("cuda") and not gpu:
        raise SNOARCLIError("--device cuda was given but no GPU is available")
    return {
        "max_steps": int(steps),
        "accelerator": "gpu" if gpu else "cpu",
        "devices": 1,
        "precision": "bf16-mixed" if gpu else "32",
    }


def _data_entry(manifest: Path, defaults, train: bool, pool=None, level=None):
    args = {
        "manifest": str(manifest),
        "segment_seconds": float(defaults.segment_seconds),
        "fs": int(defaults.fs),
    }
    if pool is not None:
        args.update(
            noise_pool=str(pool),
            noise_level=level,
            noise_snr=list(defaults.external_noise_snr),
        )
    if train:
        # Long enough epochs that validation does not dominate small corpora.
        args["num_items"] = max(1000 * int(defaults.batch_size), 1)
    else:
        args.update(deterministic=True, num_items=int(defaults.valid_max))
    return {"data_src": NOISY, "data_src_args": args}


def run_training(
    name: str,
    work: Path,
    out_dir: Path,
    model: dict,
    data: dict,
    defaults,
    device: str,
    steps: int,
    workers: Optional[int],
) -> Path:
    """Write ``<work>/<name>.yaml`` and run the ``train`` and ``export`` stages.

    Returns:
        The export directory.
    """
    from espnet3.systems.launch import launch
    from espnet3.systems.snoar.system import SNOARSystem

    nw = int(defaults.num_workers if workers is None else workers)
    cfg = {
        "exp_tag": name,
        "recipe_dir": str(work),
        "data_dir": str(work),
        "exp_dir": str(work / "exp" / name),
        "fs": int(defaults.fs),
        "dataset": {"train": [data["train"]], "valid": [data["valid"]]},
        "model": model,
        "trainer": _trainer_overrides(device, steps),
        "dataloader": {
            "train": {"batch_size": int(defaults.batch_size), "num_workers": nw},
            "valid": {
                "batch_size": int(defaults.batch_size),
                "num_workers": min(nw, 4),
            },
        },
        "export": {
            "checkpoint": str(work / "exp" / name / "last.ckpt"),
            "out_dir": str(out_dir),
        },
    }
    path = work / f"{name}.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    print(f"[snoar] {name}: {steps} steps, config {path}", file=sys.stderr)
    launch(
        SNOARSystem,
        argv=["--stages", "train", "export", "--training_config", str(path)],
    )
    return out_dir


def train_seed(mode: str, manifests, args, defaults, out_dir: Path) -> Path:
    """Train a label-free DNF or MixIT seed and export it to ``out_dir``."""
    work = Path(args.output) / "work"
    pool, level = write_noise_pool(
        args.noise,
        args.vad,
        manifests["train"],
        work,
        int(defaults.fs),
        defaults.harvest,
    )
    net = OmegaConf.to_container(defaults.net, resolve=True)
    net["separator_conf"]["n_srcs"] = 2 if mode == "dnf" else 3
    model = {
        "mode": mode,
        "net": net,
        "mix_consistency": "energy" if mode == "mixit" else None,
    }
    data = {
        "train": _data_entry(manifests["train"], defaults, True, pool, level),
        "valid": _data_entry(manifests["valid"], defaults, False, pool, level),
    }
    steps = args.seed_steps or int(defaults.seed_steps)
    return run_training(
        f"seed_{mode}",
        work,
        out_dir,
        model,
        data,
        defaults,
        args.device,
        steps,
        args.workers,
    )


# --------------------------------------------------------------------------- #
# Verbs
# --------------------------------------------------------------------------- #
def _prepare(args):
    defaults = load_cli_defaults()
    if args.fs:
        defaults.fs = int(args.fs)
    out = Path(args.output)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    manifests = write_recording_manifests(
        args.audio,
        work,
        int(defaults.fs),
        float(defaults.valid_fraction),
        int(defaults.valid_max),
    )
    return defaults, out, work, manifests


def cmd_adapt(args) -> int:
    """Self-train a seed on the recordings with SNOAR and export the result."""
    defaults, out, work, manifests = _prepare(args)
    seed = args.seed or str(defaults.seed)
    if seed in ("dnf", "mixit"):
        seed = str(train_seed(seed, manifests, args, defaults, work / f"seed_{seed}"))
    elif args.noise or args.vad:
        print(
            "[snoar] --noise/--vad only matter for --seed dnf|mixit; ignored",
            file=sys.stderr,
        )
    steps = args.steps or int(defaults.adapt_steps)
    n_gen = int(defaults.n_generations)
    model = {
        "mode": "snoar",
        "init": seed,
        "n_generations": n_gen,
        "steps_per_generation": max(1, steps // n_gen),
    }
    data = {
        "train": _data_entry(manifests["train"], defaults, True),
        "valid": _data_entry(manifests["valid"], defaults, False),
    }
    run_training(
        "adapt", work, out, model, data, defaults, args.device, steps, args.workers
    )
    print(
        f"adapted model: {out}\n  espnet snoar enhance <noisy.wav> -o <out.wav> "
        f"--model {out}",
        file=sys.stderr,
    )
    return 0


def cmd_seed(args) -> int:
    """Train a label-free DNF or MixIT seed and export it."""
    defaults, out, work, manifests = _prepare(args)
    train_seed(args.mode, manifests, args, defaults, out)
    print(
        f"seed model: {out}\n  espnet snoar adapt <recordings> -o <dir> "
        f"--seed {out}",
        file=sys.stderr,
    )
    return 0


def cmd_enhance(args) -> int:
    """Enhance one file, or every audio file of a directory."""
    import soundfile as sf

    from espnet3.systems.snoar.inference import SNOAREnhancer
    from espnet3.systems.snoar.src.data.manifest import list_audio

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    enhancer = SNOAREnhancer(
        args.model, device=device, chunk_seconds=args.chunk_seconds
    )
    fs = enhancer.fs
    src = Path(args.input)
    try:
        files = list_audio(src)
    except FileNotFoundError as e:
        raise SNOARCLIError(str(e)) from e
    dst = Path(args.output)
    single = src.is_file() and dst.suffix.lower() in (".wav", ".flac")
    for f in files:
        x, rate = sf.read(str(f), dtype="float32", always_2d=True)
        if rate != fs:
            raise SNOARCLIError(f"{f} is {rate} Hz; the model expects {fs} Hz")
        y = enhancer(x[:, 0])
        target = (
            dst
            if single
            else dst / Path(f).relative_to(src if src.is_dir() else f.parent)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), y, rate)
        print(f"wrote {target}", file=sys.stderr)
    return 0


def cmd_harvest(args) -> int:
    """Write a noise pool from the speech-free gaps of the recordings."""
    from espnet3.systems.snoar.src.data.manifest import (
        describe_recordings,
        list_audio,
        write_manifest,
    )
    from espnet3.systems.snoar.src.data.vad import harvest_noise

    defaults = load_cli_defaults()
    try:
        rows = describe_recordings(list_audio(args.audio))
    except FileNotFoundError as e:
        raise SNOARCLIError(str(e)) from e
    pool = harvest_noise(
        rows,
        vad=args.vad,
        fs=int(args.fs or defaults.fs),
        margin=args.margin,
        min_gap=args.min_gap,
    )
    n = write_manifest(pool, args.output)
    print(f"wrote {n} noise regions to {args.output}", file=sys.stderr)
    return 0


def cmd_manifest(args) -> int:
    """List recordings as a JSONL manifest."""
    from espnet3.systems.snoar.src.data.manifest import (
        describe_recordings,
        list_audio,
        write_manifest,
    )

    try:
        rows = describe_recordings(list_audio(args.audio))
    except FileNotFoundError as e:
        raise SNOARCLIError(str(e)) from e
    n = write_manifest(rows, args.output)
    print(f"wrote {n} recordings to {args.output}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_training_args(p) -> None:
    p.add_argument("audio", help="noisy recordings: a directory, a list file or a file")
    p.add_argument("-o", "--output", required=True, help="output model directory")
    p.add_argument(
        "--noise",
        help="noise-only audio (dir/list) or a pool .jsonl, " "for a DNF/MixIT seed",
    )
    p.add_argument(
        "--vad",
        help="speech labels (.rttm or Kaldi segments) whose "
        "gaps give the noise pool; default: an energy VAD",
    )
    p.add_argument("--seed-steps", type=int, help="steps of a DNF/MixIT seed run")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--workers", type=int, help="data loader workers")
    p.add_argument("--fs", type=int, help="sampling rate (default 16000)")


def add_snoar_parser(sub) -> argparse.ArgumentParser:
    """Add the ``snoar`` command group to an ``argparse`` subparsers object.

    Returns:
        The ``snoar`` parser.
    """
    p = sub.add_parser(
        "snoar",
        help="adapt a speech enhancer to unlabelled noisy audio",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verbs = p.add_subparsers(dest="snoar_command", required=True)

    a = verbs.add_parser("adapt", help="self-train a seed on the recordings (SNOAR)")
    _add_training_args(a)
    a.add_argument(
        "--seed",
        help="seed: an ESPnet2 enhancement tag, an exported "
        "directory, or dnf|mixit to train one first (default: the "
        "URGENT 2025 TF-GridNet)",
    )
    a.add_argument("--steps", type=int, help="SNOAR optimizer steps")
    a.set_defaults(func=cmd_adapt)

    s = verbs.add_parser("seed", help="train a label-free DNF or MixIT seed")
    s.add_argument("mode", choices=("dnf", "mixit"))
    _add_training_args(s)
    s.set_defaults(func=cmd_seed)

    e = verbs.add_parser("enhance", help="enhance audio with an adapted model")
    e.add_argument("input", help="a noisy file or a directory of them")
    e.add_argument("-o", "--output", required=True, help="output file or directory")
    e.add_argument("--model", required=True, help="adapted/exported model directory")
    e.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    e.add_argument(
        "--chunk-seconds",
        type=float,
        default=20.0,
        help="longest piece processed at once (overlap-added)",
    )
    e.set_defaults(func=cmd_enhance)

    h = verbs.add_parser("harvest", help="write a noise pool from VAD gaps")
    h.add_argument("audio")
    h.add_argument("-o", "--output", required=True, help="noise pool .jsonl")
    h.add_argument("--vad", default="energy", help="energy, or a .rttm/segments file")
    h.add_argument("--margin", type=float, default=0.1)
    h.add_argument("--min-gap", type=float, default=0.3)
    h.add_argument("--fs", type=int)
    h.set_defaults(func=cmd_harvest)

    m = verbs.add_parser("manifest", help="list recordings as a JSONL manifest")
    m.add_argument("audio")
    m.add_argument("-o", "--output", required=True)
    m.set_defaults(func=cmd_manifest)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``snoar <verb>`` (stand-alone entry point; ``espnet snoar`` is the same)."""
    parser = argparse.ArgumentParser(prog="espnet")
    add_snoar_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SNOARCLIError as e:
        print(f"espnet snoar: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
