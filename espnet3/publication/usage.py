"""How to use a published model, written from its system's declaration.

A system's ``Inference`` declares its inputs and outputs (:class:`Field`:
name, kind, label, optional), and the contract fixes how a model is called
- ``load``, ``model(...)``, ``model.batch``, ``model.stream`` - so the
usage a model card, a notebook or a Space shows is the same text for every
model of a system, with the tag filled in. It is written here once, from
the class alone: no weights are loaded to say how a model is used.

Examples:
    >>> from espnet3.publication.usage import build_python_usage, get_inference_class
    >>> cls = get_inference_class("s2t")
    >>> print(build_python_usage(cls, "espnet/owsm_v4_base_102M", system="s2t"))
    from espnet3.api.inference import load
    <BLANKLINE>
    model = load("espnet/owsm_v4_base_102M", system="s2t")
    result = model("speech.wav")
    ...
"""

from __future__ import annotations

import importlib
from typing import Optional

from espnet3.api.inference import SYSTEM_ALIASES, Field, InferenceAPI

# what a reader would pass for an input of each kind, in the example call
_INPUT_EXAMPLES = {"audio": '"speech.wav"', "text": '"..."', "segments": "[...]"}
# what the reader gets back, per kind of output
_OUTPUT_NOTES = {
    "audio": "an Audio: .array (samples) at .rate",
    "text": "str",
    "segments": "[{'text', 'start', 'end'}, ...] in seconds",
}


def get_inference_class(system: str) -> type[InferenceAPI]:
    """Return the ``Inference`` class of ``system``, by the name ``load`` takes.

    Args:
        system: A system name, or an alias ``load`` follows (``"s2t"``).

    Returns:
        The ``Inference`` subclass defined in ``espnet3.systems.<system>.inference``.

    Raises:
        ImportError: If the system has no ``inference`` module or it
            defines no ``Inference``.
    """
    name = SYSTEM_ALIASES.get(system, system)
    module = importlib.import_module(f"espnet3.systems.{name}.inference")
    cls = getattr(module, "Inference", None)
    if not isinstance(cls, type) or not issubclass(cls, InferenceAPI):
        raise ImportError(f"espnet3.systems.{name}.inference defines no Inference")
    return cls


def _pick_example(field: Field) -> str:
    """Return the example value written for an input of ``field``'s kind."""
    return _INPUT_EXAMPLES.get(field.kind, "...")


def build_python_usage(
    cls: type[InferenceAPI], tag: str, *, system: Optional[str] = None
) -> str:
    """Return the Python a reader runs to use the model behind ``tag``.

    Args:
        cls: The system's ``Inference`` class, whose declaration is read.
        tag: The Hub tag or directory ``load`` takes.
        system: Passed to ``load`` when the tag is not a bundle naming its
            system in ``meta.yaml`` - a checkpoint published without one.

    Returns:
        Lines of Python: the load, one call with the required inputs,
        the optional inputs and how to pass them, each output, then the
        batch and stream forms.

    Examples:
        >>> cls = get_inference_class("s2t")
        >>> text = build_python_usage(cls, "espnet/owsm_v4_base_102M", system="s2t")
        >>> text.splitlines()[0]
        'from espnet3.api.inference import load'
    """
    required = [f for f in cls.inputs if not f.optional]
    optional = [f for f in cls.inputs if f.optional]
    call = ", ".join(_pick_example(f) for f in required)
    load_args = f'"{tag}"' + (f', system="{system}"' if system else "")
    lines = [
        "from espnet3.api.inference import load",
        "",
        f"model = load({load_args})",
        f"result = model({call})",
    ]
    if optional:
        lines.append(
            "# optional inputs: " + ", ".join(f"{f.name} ({f.kind})" for f in optional)
        )
        with_optional = ", ".join(
            [call] + [f"{f.name}={_pick_example(f)}" for f in optional[:2]]
        )
        lines.append(f"# result = model({with_optional})")
    width = max(len(f'result["{f.name}"]') for f in cls.outputs)
    for f in cls.outputs:
        key = f'result["{f.name}"]'
        note = _OUTPUT_NOTES.get(f.kind, f.kind)
        lines.append(f"{key:<{width}}  # {f.label}: {note}")
    items = ", ".join(
        "{" + ", ".join(f'"{f.name}": {_pick_example(f)}' for f in required) + "}"
        for _ in range(2)
    )
    lines += [
        "",
        "# several at once, and pieces as the model produces them",
        f"model.batch([{items}])",
        "for piece in model.stream([{"
        + ", ".join(f'"{f.name}": {_pick_example(f)}' for f in required)
        + "}]):",
        "    print(piece)",
    ]
    return "\n".join(lines)


def build_usage_section(
    cls: type[InferenceAPI], tag: str, *, system: Optional[str] = None
) -> str:
    """Return the Markdown ``## Usage`` section of a model card.

    Args:
        cls: The system's ``Inference`` class, passed to :func:`build_python_usage`.
        tag: The Hub tag or directory ``load`` takes.
        system: Passed to ``load`` when the tag does not name its system.

    Returns:
        A ``## Usage`` heading followed by a fenced Python block.

    Examples:
        >>> section = build_usage_section(
        ...     get_inference_class("s2t"), "espnet/owsm_v4_base_102M", system="s2t"
        ... )
        >>> section.splitlines()[0]
        '## Usage'
    """
    body = build_python_usage(cls, tag, system=system)
    return "## Usage\n\n```python\n" + body + "\n```"
