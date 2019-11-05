import torch
from pytypes import typechecked

from espnet.utils.dynamic_import import dynamic_import


@typechecked
def build_optimizer(optim: str, model: torch.nn.Module, kwarg: dict) -> torch.optim.Optimizer:
    # Note(kamo): Don't use getattr or dynamic_import for readability and debuggability as possible

    if optim == 'Adam':
        return torch.optim.Adam(model.parameters(), **kwarg)
    elif optim == 'SGD':
        return torch.optim.SGD(model.parameters(), **kwarg)
    elif optim == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), **kwarg)
    elif optim == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), **kwarg)
    else:
        # To use any other built-in optimizer of pytorch: e.g. torch.optim.RMSprop
        if ':' not in optim:
            optimizer_class = getattr(torch.optim, optim)
        # To use custom optimizer e.g. your_module.some_file:ClassName
        else:
            optimizer_class = dynamic_import(optim)
        return optimizer_class(model.parameters(), **kwarg)
