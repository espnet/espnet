#!/usr/bin/env python3
"""
Generate systems.json for HomeQuickStart.vue from pyproject.toml.

Usage:
    python generate_systems.py path/to/pyproject.toml
    python generate_systems.py path/to/pyproject.toml -o systems.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # pip install tomli

# ── User-facing systems (exclude dev/test/doc/all) ────────────────────────────
USER_SYSTEMS = ["asr", "tts", "enh", "st", "s2st", "s2t", "spk", "speechlm"]

SYSTEM_META = {
    "asr": {
        "name": "Automatic Speech Recognition",
        "desc": (
            "End-to-end ASR with CTC, attention, and hybrid decoding."
            " Includes WER evaluation and CTC segmentation."
        ),
    },
    "tts": {
        "name": "Text-to-Speech",
        "desc": (
            "Neural TTS with support for multiple languages and frontend"
            " systems including G2P and pitch extraction."
        ),
    },
    "enh": {
        "name": "Speech Enhancement",
        "desc": (
            "Speech separation and enhancement. Includes BSS evaluation"
            " metrics like SI-SDR and fast-bss-eval."
        ),
    },
    "st": {
        "name": "Speech Translation",
        "desc": (
            "Speech-to-text translation across languages." " Evaluation via BLEU score."
        ),
    },
    "s2st": {
        "name": "Speech-to-Speech Translation",
        "desc": (
            "Direct speech-to-speech translation with S3PRL integration"
            " for self-supervised representations."
        ),
    },
    "s2t": {
        "name": "Speech-to-Text (general)",
        "desc": (
            "General speech-to-text framework covering tasks beyond" " standard ASR."
        ),
    },
    "spk": {
        "name": "Speaker",
        "desc": (
            "Speaker diarization, identification, and embedding extraction."
            " Supports x-vector, ECAPA-TDNN and more."
        ),
    },
    "speechlm": {
        "name": "Speech LM",
        "desc": (
            "LLM integration for speech tasks. Includes Transformers,"
            " torchtitan (Linux/CUDA), and Liger kernel."
        ),
    },
}


def parse_pkg(dep: str) -> dict:
    """Parse a PEP 508 dependency string into {name, src, note}."""
    # strip environment markers like "; sys_platform == 'linux'"
    note = None
    if ";" in dep:
        dep, marker = dep.split(";", 1)
        marker = marker.strip()
        if "linux" in marker:
            note = "Linux only"
        elif "windows" in marker:
            note = "Windows only"
        elif "darwin" in marker:
            note = "macOS only"

    dep = dep.strip()

    # git+ URL  →  extract package name from the URL path
    if "@ git+" in dep:
        name = dep.split("@")[0].strip()
        src = "github"
    elif dep.startswith("git+"):
        # bare git URL without a name prefix (rare)
        name = re.split(r"[/#@]", dep.rstrip("/"))[-1]
        src = "github"
    else:
        # normal pypi dep: strip version specifiers
        name = re.split(r"[><=!~\[;]", dep)[0].strip()
        src = "pypi"

    result = {"name": name, "src": src}
    if note:
        result["note"] = note
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate systems.json from pyproject.toml"
    )
    parser.add_argument("pyproject", type=Path, help="Path to pyproject.toml")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    with open(args.pyproject, "rb") as f:
        data = tomllib.load(f)

    optional_deps = data.get("project", {}).get("optional-dependencies", {})

    systems = []
    for key in USER_SYSTEMS:
        if key not in optional_deps:
            print(
                f"Warning: system '{key}' not found in pyproject.toml",
                file=sys.stderr,
            )
            continue

        meta = SYSTEM_META.get(key, {"name": key, "desc": ""})
        pkgs = [parse_pkg(dep) for dep in optional_deps[key]]

        systems.append(
            {
                "key": key,
                "name": meta["name"],
                "desc": meta["desc"],
                "group": f"espnet[{key}]",
                "pkgs": pkgs,
            }
        )

    out = json.dumps(systems, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(out)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
