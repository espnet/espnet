"""The preprocessor arguments an S2T checkpoint was trained with."""

import inspect
from argparse import Namespace
from typing import Any, Dict, Type

from espnet2.tasks.abs_task import AbsTask


def train_preprocessor_conf(task: Type[AbsTask], args: Namespace) -> Dict[str, Any]:
    """The preprocessor's arguments as training used them, defaults included.

    A training config stores only the arguments that were set, so one that
    left `fs` at the preprocessor's default - the OWSM-CTC configs do - has
    no `fs` to read. Filling the gaps from the signature of the class that
    was trained with gives the values it actually ran with, rather than a
    second copy of its defaults at every place that reads them.

    Args:
        task: the task that trained the checkpoint, whose `preprocessor`
            choice names the class.
        args: the training arguments read back from its config.

    Returns:
        The preprocessor's arguments; the ones the config sets take
        precedence over the class defaults.
    """
    conf = dict(getattr(args, "preprocessor_conf", None) or {})
    name = getattr(args, "preprocessor", None)
    choices = {c.name: c for c in task.class_choices_list}.get("preprocessor")
    if name is None or choices is None:
        return conf
    defaults = {
        key: param.default
        for key, param in inspect.signature(choices.get_class(name)).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    return {**defaults, **conf}
