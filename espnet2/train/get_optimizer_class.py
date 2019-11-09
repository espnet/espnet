import torch
from typeguard import check_argument_types, typechecked
from typing import Type

from espnet.utils.dynamic_import import dynamic_import


@typechecked
def get_optimizer_class(optim: str) -> Type[torch.optim.Optimizer]:
    # Note(kamo): Don't use getattr or dynamic_import
    # for readability and debuggability as possible

    if optim.lower() == 'adam':
        return torch.optim.Adam
    elif optim.lower() == 'sgd':
        return torch.optim.SGD
    elif optim.lower() == 'adadelta':
        return torch.optim.Adadelta
    else:
        # To use any other built-in optimizer of pytorch, e.g. RMSprop
        if ':' not in optim:
            optimizer_class = getattr(torch.optim, optim)
        # To use custom optimizer e.g. your_module.some_file:ClassName
        else:
            optimizer_class = dynamic_import(optim)
