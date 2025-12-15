"""Wrapper to step multiple optimizers at once in Lightning."""

# This script is copied from
# https://github.com/Lightning-AI/pytorch-lightning/issues/3346#issuecomment-1478556073

from typing import Any, Callable, Dict, Iterable, List, Union

import torch
from typeguard import typechecked


class MultipleOptim(torch.optim.Optimizer):
    """Wrapper to step multiple optimizers at once in Lightning.

    Wrapper around multiple optimizers that should be stepped together at a single
    time. This is a hack to avoid PyTorch Lightning calling ``training_step`` once
    for each optimizer, which increases training time and is not always necessary.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Args:
        optimizers: list of optimizers

    """

    @typechecked
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        """Initialize MultipleOptim object."""
        self.optimizers = optimizers

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        """Return the combined state for each optimizer in ``self.optimizers``."""
        return {
            key: value
            for optimizer in self.optimizers
            for key, value in optimizer.state.items()
        }

    @property
    def param_groups(self) -> List[Dict[str, Union[torch.Tensor, float, bool, Any]]]:
        """Return the combined parameters for each optimizer in ``self.optimizers``."""
        return [
            element
            for optimizer in self.optimizers
            for element in optimizer.param_groups
        ]

    @property
    def defaults(self) -> Dict[str, torch.Tensor]:
        """Return the combined defaults for each optimizer in ``self.optimizers``."""
        return {
            key: value
            for optimizer in self.optimizers
            for key, value in optimizer.defaults.items()
        }

    def __getstate__(self) -> List[torch.optim.Optimizer]:
        """Return optimizers."""
        return self.optimizers

    def __setstate__(self, optimizers: List[torch.optim.Optimizer]) -> None:
        """Set optimizers."""
        self.optimizers = optimizers

        # call remaining lines of the ``torch.optim.Optimizer.__setstate__`` method.
        # copied from:
        # https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()  # To support multiprocessing pickle/unpickle.
            optimizer.defaults.setdefault("differentiable", False)

    def __repr__(self) -> str:
        """Return the string representation of the MultipleOptim object."""
        repr_str = (
            f"``{self.__class__.__name__}`` "
            + f"containing {len(self.optimizers)} optimizers:\n"
        )

        for optimizer in self.optimizers:
            repr_str += "\n" + optimizer.__repr__()

        return repr_str

    def _hook_for_profile(self) -> None:
        """Call ``_hook_for_profile`` for each optimizer in ``self.optimizers``."""
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()

    def state_dict(
        self,
    ) -> List[
        Dict[
            str,
            Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, float, bool, Any]]]],
        ]
    ]:
        """Return the state of the optimizer as a dictionary.

        It contains two entries:

            * ``state`` - a dict holding current optimization state.
                Its content differs between optimizer classes.
            * ``param_groups`` - a list containing all parameter groups
                where each parameter group is a dict

        """
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(
        self,
        state_dict: List[
            Dict[
                str,
                Union[
                    torch.Tensor, List[Dict[str, Union[torch.Tensor, float, bool, Any]]]
                ],
            ]
        ],
    ) -> None:
        """Load the optimizer state.

        Args:
        state_dict (dict): Optimizer state. Should be an object returned from a call to
            ``state_dict()``

        """
        for state, optimizer in zip(state_dict, self.optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Set the gradients of all optimized ``torch.Tensor``s to zero.

        Args:
            set_to_none: Instead of setting to zero, set the grads to ``None``.
                This will in general have lower memory footprint, and can modestly
                improve performance. However, it changes certain behaviors. For example:

                    1. When the user tries to access a gradient and perform manual ops
                        on it, a ``None`` attribute or a ``torch.Tensor`` full of ``0``s
                        will behave differently.

                    2. If the user requests ``zero_grad(set_to_none=True)`` followed by
                        a backward pass, ``.grad``s are guaranteed to be ``None`` for
                        params that did not receive a gradient.

                    3. ``torch.optim`` optimizers have a different behavior if the
                        gradient is ``0`` or ``None`` (in one case it does the step
                        with a gradient of ``0`` and in the other it skips the step
                        altogether).

        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] = None) -> torch.Tensor:
        """Perform a single optimization step.

        Args:
            closure: function
                A closure that reevaluates the model and returns the loss. Optional
                for most optimizers.

        Notes:
            Unless otherwise specified, this function should not modify the ``.grad``
            field of the parameters.

        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss
