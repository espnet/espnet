"""Check that a streaming object keeps bounded memory however long it runs.

A component that is fed audio chunk by chunk, such as a VAD in a dialogue
system, may run for hours. If it appends something on every call and never
drops it, its memory grows with the length of the session, and no test that
runs it for a few seconds notices. These helpers measure the memory an object
holds and fail a test when it keeps growing.

    from test.memory_growth import assert_memory_bounded

    vad = FrontiaVADModel(...)
    chunk = np.zeros(16000, dtype=np.int16)
    assert_memory_bounded(vad, lambda: vad.forward(chunk, 16000), steps=600)

retained_nbytes() counts what is reachable from the object's attributes: the
buffer behind each numpy array (the whole base array of a view, which a view
keeps alive), the storage of each torch tensor, bytes and str, and the size of
every list, tuple, dict, set and deque, followed into their items. Modules,
classes and functions are not followed.
"""

import sys
import types
from collections import deque
from typing import Callable, Dict, List

import numpy as np
import torch

_NOT_FOLLOWED = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)


def retained_nbytes(obj) -> int:
    """Return the bytes of the data reachable from obj, each buffer once."""
    seen = set()
    total = 0
    stack = [obj]
    while stack:
        x = stack.pop()
        if id(x) in seen or isinstance(x, _NOT_FOLLOWED):
            continue
        seen.add(id(x))
        if isinstance(x, np.ndarray):
            # A view keeps its whole base array alive, so count the base.
            base = x
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in seen or base is x:
                seen.add(id(base))
                total += base.nbytes
        elif isinstance(x, torch.Tensor):
            storage = x.untyped_storage()
            key = ("storage", storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(x, (bytes, bytearray, str)):
            total += sys.getsizeof(x)
        elif isinstance(x, (list, tuple, set, frozenset, deque)):
            total += sys.getsizeof(x)
            stack.extend(x)
        elif isinstance(x, dict):
            total += sys.getsizeof(x)
            stack.extend(x.keys())
            stack.extend(x.values())
        else:
            if hasattr(x, "__dict__"):
                stack.append(vars(x))
            for name in getattr(type(x), "__slots__", ()):
                if hasattr(x, name):
                    stack.append(getattr(x, name))
    return total


def retained_nbytes_by_attribute(obj, depth: int = 2) -> Dict[str, int]:
    """Return retained_nbytes of each attribute of obj, down to `depth` levels.

    Keys are attribute paths such as "stream.probs". A buffer reachable from
    two paths counts in both, so the values do not add up to the total; they
    say where memory is held.
    """
    sizes = {}

    def walk(x, prefix, level):
        if not hasattr(x, "__dict__") or isinstance(x, _NOT_FOLLOWED):
            return
        for name, value in vars(x).items():
            path = f"{prefix}{name}"
            sizes[path] = retained_nbytes(value)
            if level < depth:
                walk(value, path + ".", level + 1)

    walk(obj, "", 1)
    return sizes


def assert_memory_bounded(
    obj,
    step: Callable[[], object],
    steps: int,
    warmup: int = 10,
    samples: int = 20,
    slack_bytes: int = 0,
) -> List[int]:
    """Run step() many times and fail if obj's retained memory keeps growing.

    After warmup steps, obj is measured `samples` times, evenly over `steps`
    steps. The test fails when the smallest measurement of the second half
    exceeds the largest of the first half by more than slack_bytes. Memory
    that grows with the number of steps raises its floor above everything it
    held earlier. Memory that goes up and down within a bound, as a buffer
    that holds the current turn does, keeps its floor and passes.

    Args:
        obj: The object whose memory is measured, e.g. a streaming VAD.
        step: Feeds obj one chunk.
        steps: The number of steps measured after the warm-up.
        warmup: Steps before the first measurement, for buffers that fill to
            their bound at the start.
        samples: The number of measurements, at least 4.
        slack_bytes: The growth allowed between the two halves.

    Returns:
        The measurements, in bytes.
    """
    assert samples >= 4
    for _ in range(warmup):
        step()
    sizes = []
    before = retained_nbytes_by_attribute(obj)
    every = max(1, steps // samples)
    for i in range(steps):
        step()
        if (i + 1) % every == 0:
            sizes.append(retained_nbytes(obj))
    half = len(sizes) // 2
    first_peak, second_floor = max(sizes[:half]), min(sizes[half:])
    if second_floor > first_peak + slack_bytes:
        after = retained_nbytes_by_attribute(obj)
        grown = sorted(((after[k] - before.get(k, 0), k) for k in after), reverse=True)[
            :5
        ]
        raise AssertionError(
            f"retained memory grows with the stream: at least {second_floor} "
            f"bytes in the second half, at most {first_peak} in the first; "
            f"measurements {sizes}; grown most: "
            + ", ".join(f"{k} +{n}" for n, k in grown if n > 0)
        )
    return sizes
