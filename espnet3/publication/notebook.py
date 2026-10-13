r"""A demo notebook for a published model, written from its system's declaration.

The notebooks in https://github.com/espnet/notebook/tree/master/Demos are
one page per task: install, an input, the model's call, its outputs, and
where to go next. Every part but the prose comes from the system's
``Inference`` declaration - the same :func:`espnet3.publication.usage`
lines a model card shows - so a notebook for a new model is this module
run with its tag, not a page written by hand.

The page is written with ``nbformat``, the ``notebook`` extra
(``pip install "espnet[notebook]"``). It keeps the rules that repository's
``tools/check_layout.py`` holds a demo to: an espnet release pinned in a
``%pip install`` line, and the badge of the workflow that runs it.

Examples:
    >>> from espnet3.publication.notebook import build_demo_notebook, write_notebook
    >>> from espnet3.publication.usage import get_inference_class
    >>> page = build_demo_notebook(
    ...     get_inference_class("s2t"), "espnet/owsm_v4_base_102M", system="s2t",
    ...     name="asr_demo", title="Speech recognition with OWSM v4",
    ...     inputs={"language": "eng"}, espnet_version="202610.post2",
    ... )
    >>> write_notebook(page, "Demos/asr_demo.ipynb")  # doctest: +SKIP

    The same from a shell::

        python -m espnet3.publication.notebook espnet/owsm_v4_base_102M \\
            --system s2t --title "Speech recognition with OWSM v4" \\
            --input language=eng -o Demos/asr_demo.ipynb
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from espnet3.api.inference import Field, InferenceAPI
from espnet3.publication.usage import (
    build_call_lines,
    build_more_lines,
    get_inference_class,
)

#: A sentence of read English, 16 kHz, in the espnet repository.
SAMPLE_AUDIO = (
    "https://github.com/espnet/espnet/raw/master/test_utils/ctc_align_test.wav"
)
REPO = "https://github.com/espnet/notebook"


def _read_first_sentence(cls: type) -> str:
    """Return the first line of the class's docstring, or nothing."""
    return (cls.__doc__ or "").strip().split("\n", 1)[0]


def _collect_audio_paths(
    cls: type[InferenceAPI], inputs: Mapping[str, Any]
) -> list[str]:
    """Return the files the call reads: required audio, and optional audio given."""
    return [
        inputs.get(f.name, "speech.wav")
        for f in cls.inputs
        if f.kind == "audio" and (not f.optional or f.name in inputs)
    ]


def _show_outputs(outputs: Sequence[Field]) -> str:
    """Python that shows each output the way its kind is looked at."""
    lines = []
    if any(f.kind == "audio" for f in outputs):
        lines += ["from IPython.display import Audio, display", ""]
    for f in outputs:
        key = f'result["{f.name}"]'
        if f.kind == "text":
            lines.append(f"print({key})")
        elif f.kind == "segments":
            lines += [
                f"for segment in {key}:",
                "    print(f\"{segment['start']:7.2f} {segment['end']:7.2f}  "
                "{segment['text']}\")",
            ]
        else:  # audio
            lines.append(f"display(Audio({key}.array, rate={key}.rate))")
    return "\n".join(lines)


def build_demo_notebook(
    cls: type[InferenceAPI],
    tag: str,
    *,
    name: str,
    title: str,
    system: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    espnet_version: Optional[str] = None,
    sample_audio: str = SAMPLE_AUDIO,
    space: Optional[str] = None,
) -> nbformat.NotebookNode:
    """Return the Demos notebook for ``tag``, built from its system's declaration.

    Args:
        cls: The system's ``Inference`` class, whose declaration is read.
        tag: The Hub tag ``load`` takes.
        name: The notebook's file stem, ``<task>_demo`` in that repository;
            its Colab link and workflow badge are written from it.
        title: The page's heading.
        system: Passed to ``load`` for a checkpoint published without a
            bundle, as the card's usage passes it.
        inputs: Values written in the call: a path for an audio input (the
            sample is downloaded to it), the text for a text input, and the
            optional inputs this page is about (``{"task": "st_deu"}`` for
            a translation page on a model that also transcribes).
        espnet_version: The release the install cell pins; the one
            installed here when not given, and no pin without one.
        sample_audio: What the audio inputs are downloaded from.
        space: The model's Hugging Face Space, linked at the end if given.

    Returns:
        An nbformat 4 notebook: install, the sample recording when the model
        takes audio, the call, the outputs, the batch and stream forms, and
        where next.

    Examples:
        >>> from espnet3.publication.usage import get_inference_class
        >>> page = build_demo_notebook(
        ...     get_inference_class("s2t"), "espnet/owsm_v4_base_102M",
        ...     name="asr_demo", title="Speech recognition with OWSM v4",
        ...     system="s2t", espnet_version="202610.post2",
        ... )
        >>> page.nbformat
        4
    """
    given = dict(inputs or {})
    if espnet_version is None:
        from espnet import __version__

        espnet_version = __version__ if __version__[:1].isdigit() else None
    pin = f'"espnet=={espnet_version}"' if espnet_version else "espnet"
    paths = _collect_audio_paths(cls, given)
    run = " ".join(
        ["espnet", "run", shlex.quote(tag)]
        + (["--system", shlex.quote(system)] if system else [])
        + [
            (
                f"{f.name}=" + shlex.quote(str(given[f.name]))
                if f.name in given
                else f"{f.name}=" + (paths[0] if f.kind == "audio" else "...")
            )
            for f in cls.inputs
            if not f.optional or f.name in given
        ]
    )
    cells = [
        new_markdown_cell(
            f"# {title}\n\n"
            "[![Open In Colab](https://colab.research.google.com/assets/"
            f"colab-badge.svg)](https://colab.research.google.com/github/espnet/"
            f"notebook/blob/master/Demos/{name}.ipynb) "
            f"[![{name}]({REPO}/actions/workflows/{name}.yml/badge.svg)]"
            f"({REPO}/actions/workflows/{name}.yml)\n\n"
            f"{_read_first_sentence(cls)} The model is "
            f"[`{tag}`](https://huggingface.co/{tag}); the first run downloads "
            "it, and it is cached after that.\n\n"
            "This page is written from what the model declares - its inputs "
            "and outputs - so the same cells run for any model of its system."
        ),
        new_markdown_cell("## Install"),
        new_code_cell(f"%pip install -q {pin} espnet_model_zoo"),
    ]
    if paths:
        cells += [
            new_markdown_cell("## A recording to work with\n\nAny wav will do."),
            new_code_cell(
                "\n".join(f"!wget -q -O {path} {sample_audio}" for path in paths)
                + "\n\nfrom IPython.display import Audio, display\n\n"
                + "\n".join(f'display(Audio("{path}"))' for path in paths)
            ),
        ]
    cells += [
        new_markdown_cell(
            "## Run it\n\n`load` reads the model; calling it with the inputs it "
            "declares returns its outputs by name."
        ),
        new_code_cell(
            "\n".join(build_call_lines(cls, tag, system=system, inputs=given))
        ),
        new_markdown_cell("## What came back"),
        new_code_cell(_show_outputs(cls.outputs)),
        new_markdown_cell(
            "## Several at once, and as it goes\n\n`batch` takes a list of "
            "inputs; `stream` yields pieces as the model produces them."
        ),
        new_code_cell("\n".join(build_more_lines(cls, inputs=given))),
        new_markdown_cell(
            "## Where next\n\n"
            f"- **From the terminal**: `pip install espnet`, then `{run}`\n"
            + (f"- **In the browser**: the [Space]({space})\n" if space else "")
            + f"- **The model card**: https://huggingface.co/{tag}"
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )


def write_notebook(notebook: nbformat.NotebookNode, path: str | Path) -> Path:
    """Write ``notebook`` as an ``.ipynb`` file and return its path.

    Args:
        notebook: A notebook from :func:`build_demo_notebook`.
        path: Where to write it; its parent directories are created.

    Returns:
        The path written.

    Examples:
        >>> write_notebook(page, "Demos/asr_demo.ipynb")  # doctest: +SKIP
        PosixPath('Demos/asr_demo.ipynb')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write one demo notebook from a tag and its system."""
    parser = argparse.ArgumentParser(
        prog="python -m espnet3.publication.notebook",
        description="write a Demos/ notebook for a published model",
    )
    parser.add_argument("tag", help="the Hub tag load takes")
    parser.add_argument("--system", help="the system name, for a tag without a bundle")
    parser.add_argument("--title", required=True, help="the page's heading")
    parser.add_argument(
        "-o", "--output", required=True, help="where to write, Demos/<task>_demo.ipynb"
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a value written in the call (repeatable)",
    )
    parser.add_argument("--espnet-version", help="the release to pin")
    parser.add_argument("--sample-audio", default=SAMPLE_AUDIO, help="a wav URL")
    parser.add_argument("--space", help="the model's Space, linked at the end")
    args = parser.parse_args(argv)
    inputs = {}
    for pair in args.input:
        name, sep, value = pair.partition("=")
        if not sep:
            parser.error(f"--input {pair!r}: give NAME=VALUE")
        inputs[name] = value
    system = args.system
    if not system:
        # a bundle names its system in meta.yaml; no weights are built for it
        from espnet3.api.inference.loading import locate_pack, read_meta

        system = read_meta(locate_pack(args.tag)).get("system")
        if not system:
            parser.error(f"{args.tag} names no system in its meta.yaml; pass --system")
    cls = get_inference_class(system)
    page = build_demo_notebook(
        cls,
        args.tag,
        name=Path(args.output).stem,
        title=args.title,
        system=args.system,
        inputs=inputs,
        espnet_version=args.espnet_version,
        sample_audio=args.sample_audio,
        space=args.space,
    )
    print(write_notebook(page, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
