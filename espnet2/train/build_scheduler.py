from abc import ABC
from typing import Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytypes import typechecked

from espnet.utils.dynamic_import import dynamic_import


# If you need to define custom scheduler, please inherit these classes
class AbsBatchScheduler(ABC):
    def step(self, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


class AbsEpochScheduler(ABC):
    def step(self, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


class AbsValEpochScheduler(ABC):
    def step(self, val, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


# Create alias type to check the type
# Note(kamo): Currently PyTorch doesn't provide the base class to distinguish these classes.
ValEpochScheduler = Union[AbsEpochScheduler, ReduceLROnPlateau]
EpochScheduler = Union[AbsEpochScheduler, ValEpochScheduler,
                       LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR]
# FIXME(kamo): BatchScheduler is a confusing name. Please give me an idea.
BatchScheduler = Union[AbsBatchScheduler, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts]


@typechecked
def build_epoch_scheduler(scheduler: str, optimizer: torch.optim.Optimizer, kwargs) -> EpochScheduler:
    """PyTorch schedulers control optim-parameters at the end of each epochs

    EpochScheduler:
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     scheduler.step()

    ValEpochScheduler:
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val = validate(...)
        >>>     scheduler.step(val)
    """
    # Note(kamo): Don't use getattr or dynamic_import for readability and debuggability as possible

    if scheduler == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler == 'LambdaLR':
        return LambdaLR(optimizer, **kwargs)
    elif scheduler == 'StepLR':
        return StepLR(optimizer, **kwargs)
    elif scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, **kwargs)
    elif scheduler == 'ExponentialLR':
        return ExponentialLR(optimizer, **kwargs)
    elif scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        # To use custom scheduler e.g. your_module.some_file:ClassName
        scheduler_class = dynamic_import(scheduler)
        return scheduler_class(optimizer, **kwargs)


@typechecked
def build_batch_scheduler(scheduler: str, optimizer: torch.optim.Optimizer, kwargs: dict) -> BatchScheduler:
    """PyTorch schedulers control optim-parameters after every updating

    BatchScheduler:
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """
    # Note(kamo): Don't use getattr or dynamic_import for readability and debuggability as possible

    if scheduler == 'CyclicLR':
        return CyclicLR(optimizer, **kwargs)
    elif scheduler == 'OneCycleLR':
        return OneCycleLR(optimizer, **kwargs)
    elif scheduler == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(optimizer, **kwargs)
    else:
        # To use custom scheduler e.g. your_module.some_file:ClassName
        scheduler_class = dynamic_import(scheduler)
        return scheduler_class(optimizer, **kwargs)
