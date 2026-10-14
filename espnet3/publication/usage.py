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
import json
from typing import Any, Mapping, Optional

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


def _format_literal(value: Any) -> str:
    """Return ``value`` as a reader would type it: strings in double quotes."""
    return (
        json.dumps(value, ensure_ascii=False) if isinstance(value, str) else repr(value)
    )


def _write_value(field: Field, given: Mapping[str, Any]) -> str:
    """Return the value written for ``field``: the given one, else an example."""
    if field.name in given:
        return _format_literal(given[field.name])
    return _pick_example(field)


def build_call_lines(
    cls: type[InferenceAPI],
    tag: str,
    *,
    system: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return the load, one call, and the outputs, a line each.

    Args:
        cls: The system's ``Inference`` class, whose declaration is read.
        tag: The Hub tag or directory ``load`` takes.
        system: Passed to ``load`` when the tag is not a bundle naming its
            system in ``meta.yaml`` - a checkpoint published without one.
        inputs: Values to write in the call instead of the examples: a
            path for an audio input, the text for a text input; an optional
            input named here is passed, the others are listed in a comment.

    Returns:
        Lines of Python: the load, the call, the optional inputs in a comment,
        then one line per output.
    """
    given = dict(inputs or {})
    required = [f for f in cls.inputs if not f.optional]
    passed = [f for f in cls.inputs if f.optional and f.name in given]
    left = [f for f in cls.inputs if f.optional and f.name not in given]

    call = ", ".join(
        [_write_value(f, given) for f in required]
        + [f"{f.name}={_write_value(f, given)}" for f in passed]
    )
    load_args = _format_literal(tag) + (
        f", system={_format_literal(system)}" if system else ""
    )
    lines = [
        "from espnet3.api.inference import load",
        "",
        f"model = load({load_args})",
        f"result = model({call})",
    ]
    if left:
        lines.append(
            "# optional inputs: " + ", ".join(f"{f.name} ({f.kind})" for f in left)
        )
        with_optional = ", ".join(
            [call] + [f"{f.name}={_pick_example(f)}" for f in left[:2]]
        )
        lines.append(f"# result = model({with_optional})")
    width = max(len(f'result["{f.name}"]') for f in cls.outputs)
    for f in cls.outputs:
        key = f'result["{f.name}"]'
        note = _OUTPUT_NOTES.get(f.kind, f.kind)
        lines.append(f"{key:<{width}}  # {f.label}: {note}")
    return lines


def build_more_lines(
    cls: type[InferenceAPI], *, inputs: Optional[Mapping[str, Any]] = None
) -> list[str]:
    """Return the batch and stream forms of the call, a line each.

    An item carries the required inputs and the optional ones ``inputs``
    gives, as the call does: a page about translation translates in batch.

    Args:
        cls: The system's ``Inference`` class, whose declaration is read.
        inputs: Values for optional inputs, written into each item.

    Returns:
        Two lines: the batch call and the stream loop, each with an item.
    """
    given = dict(inputs or {})
    fields = [f for f in cls.inputs if not f.optional or f.name in given]
    item = (
        "{" + ", ".join(f'"{f.name}": {_write_value(f, given)}' for f in fields) + "}"
    )
    return [
        "# several at once, and pieces as the model produces them",
        f"model.batch([{item}, {item}])",
        f"for piece in model.stream([{item}]):",
        "    print(piece)",
    ]


def build_python_usage(
    cls: type[InferenceAPI],
    tag: str,
    *,
    system: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the Python a reader runs to use the model behind ``tag``.

    :func:`build_call_lines` - the load, one call with the required inputs,
    the optional inputs and how to pass them, each output - then a blank line
    and :func:`build_more_lines`, the batch and stream forms. The arguments are
    theirs.

    Args:
        cls: The system's ``Inference`` class, whose declaration is read.
        tag: The Hub tag or directory ``load`` takes.
        system: Passed to ``load`` when the tag does not name its system.
        inputs: Values to write in the call instead of the examples.

    Returns:
        Lines of Python, joined with newlines.

    Examples:
        >>> cls = get_inference_class("s2t")
        >>> text = build_python_usage(cls, "espnet/owsm_v4_base_102M", system="s2t")
        >>> text.splitlines()[0]
        'from espnet3.api.inference import load'
    """
    return "\n".join(
        build_call_lines(cls, tag, system=system, inputs=inputs)
        + [""]
        + build_more_lines(cls, inputs=inputs)
    )


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
