"""How the output slots of a network are read, for each kind of seed.

One ``readout`` name says what a trained network's slots mean, and from it
follow the three reads SNOAR needs. This replaces the reference's arm
registry (``bdnf/arms.py``), whose job was to keep these reads consistent.

``dnf``
    Two slots ``(s_noisy, n)`` trained ``x1 + n2 -> (x1, n2)``. Speech is the
    eq.-14 projection ``s_noisy - c n``.
``mixit``
    Three slots ``(s, n_a, n_b)`` trained with the constrained MixIT loss.
    Speech is slot 0; noise is ``n_a + n_b``.
``raw``
    Slot 0 is a direct estimate of the (reverberant) speech image, e.g. a
    supervised model or a SNOAR student of one. Any other slot is noise.
"""

from __future__ import annotations

from typing import Tuple

import torch

from espnet3.systems.snoar.src.ops import align_scale, decontaminate, project

READOUTS = ("dnf", "mixit", "raw")
_SLOTS = {"dnf": (2,), "mixit": (3,), "raw": (1, 2)}


def check_readout(readout: str, num_outputs: int) -> None:
    """Raise if ``readout`` cannot read a network with ``num_outputs`` slots.

    Raises:
        ValueError: If the readout is unknown or the slot count does not fit.

    Examples:
        >>> check_readout("dnf", 2)
        >>> check_readout("dnf", 1)
        Traceback (most recent call last):
        ...
        ValueError: readout 'dnf' needs 2 output slot(s), the network has 1
    """
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    if num_outputs not in _SLOTS[readout]:
        want = " or ".join(str(n) for n in _SLOTS[readout])
        raise ValueError(
            f"readout {readout!r} needs {want} output slot(s), "
            f"the network has {num_outputs}"
        )


def read_teacher(
    out: torch.Tensor,
    x: torch.Tensor,
    readout: str,
    tau_rel: float = 1e-3,
    purify_noise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn teacher outputs on ``x`` into (speech, noise) pseudo-labels.

    Mirrors ``_teacher_est`` of the reference. The speech label is always
    pulled into the gain of ``x`` with ``align_scale``.

    Args:
        out: Teacher outputs ``(B, J, T)``.
        x: Teacher input ``(B, T)``.
        readout: One of :data:`READOUTS`.
        tau_rel: Guard of the projection (``dnf`` and ``mixit``).
        purify_noise: Remove the speech component from the noise label
            (``decontaminate``) for ``dnf``/``mixit``. ``raw`` always uses the
            mixture-consistent residual ``x - speech`` instead, because a
            supervised noise head sits in an unrelated SI-SDR gain.

    Returns:
        ``(speech, noise)``, each ``(B, T)``.
    """
    check_readout(readout, out.shape[1])
    if readout == "raw":
        sp = align_scale(out[:, 0], x)
        return sp, x - sp
    noise = out[:, 1] if readout == "dnf" else out[:, 1] + out[:, 2]
    sp = align_scale(project(out[:, 0], noise, tau_rel=tau_rel), x)
    return sp, (decontaminate(noise, sp) if purify_noise else noise)


def read_student(
    out: torch.Tensor, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the student's (speech, noise) estimates for the remix losses.

    One slot: noise is the residual ``x - speech``. Two slots: slot 1. Three
    slots: ``slot 1 + slot 2``, MixIT's own noise convention.
    """
    j = out.shape[1]
    if j == 1:
        return out[:, 0], x - out[:, 0]
    if j == 2:
        return out[:, 0], out[:, 1]
    if j == 3:
        return out[:, 0], out[:, 1] + out[:, 2]
    raise ValueError(f"cannot read a student with {j} output slots")


def read_speech(
    out: torch.Tensor, x: torch.Tensor, readout: str, tau_rel: float = 1e-3
) -> torch.Tensor:
    """Return the enhanced speech for inference, at the gain of ``x``.

    ``dnf`` projects; ``mixit`` and ``raw`` read slot 0. The result is
    scaled with ``align_scale`` so that its level follows the input, which
    also keeps chunked long-form inference free of gain jumps.
    """
    check_readout(readout, out.shape[1])
    if readout == "dnf":
        speech = project(out[:, 0], out[:, 1], tau_rel=tau_rel)
    else:
        speech = out[:, 0]
    return align_scale(speech, x)
