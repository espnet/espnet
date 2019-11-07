from abc import ABC
from typing import Union, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import CyclicLR
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytypes import typechecked

from espnet.utils.dynamic_import import dynamic_import


# FIXME(kamo): EpochScheduler, BatchScheduler is a confusing name. Please give me an idea.


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


class AbsValEpochScheduler(AbsBatchScheduler):
    def step(self, val, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass


# Create alias type to check the type
# Note(kamo): Currently PyTorch doesn't provide the base class
# to judge these classes.
AbsValEpochScheduler.register(ReduceLROnPlateau)
AbsEpochScheduler.register(LambdaLR)
AbsEpochScheduler.register(StepLR)
AbsEpochScheduler.register(MultiStepLR)
AbsEpochScheduler.register(ExponentialLR)
AbsEpochScheduler.register(CosineAnnealingLR)

# AbsBatchScheduler.register(CyclicLR)
# AbsBatchScheduler.register(OneCycleLR)
# AbsBatchScheduler.register(CosineAnnealingWarmRestarts)


@typechecked
def build_epoch_scheduler(optimizer: torch.optim.Optimizer,
                          scheduler: Optional[str], kwargs) \
        -> Optional[AbsEpochScheduler]:
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
    # Note(kamo): Don't use getattr or dynamic_import
    # for readability and debuggability as possible

    if scheduler is None:
        return None

    if scheduler.lower() == 'reducelronplateau':
        return ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler.lower() == 'lambdalr':
        return LambdaLR(optimizer, **kwargs)
    elif scheduler.lower() == 'steplr':
        return StepLR(optimizer, **kwargs)
    elif scheduler.lower() == 'multisteplr':
        return MultiStepLR(optimizer, **kwargs)
    elif scheduler.lower() == 'exponentiallr':
        return ExponentialLR(optimizer, **kwargs)
    elif scheduler.lower() == 'cosineannealinglr':
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        # To use custom scheduler e.g. your_module.some_file:ClassName
        scheduler_class = dynamic_import(scheduler)
        return scheduler_class(optimizer, **kwargs)


@typechecked
def build_batch_scheduler(optimizer: torch.optim.Optimizer,
                          scheduler: Optional[str], kwargs: dict) \
        -> Optional[AbsBatchScheduler]:
    """PyTorch schedulers control optim-parameters after every updating

    BatchScheduler:
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """
    # Note(kamo): Don't use getattr or dynamic_import
    # for readability and debuggability as possible

    if scheduler is None:
        return None

    if scheduler.lower() == 'cycliclr':
        return CyclicLR(optimizer, **kwargs)
    elif scheduler.lower() == 'onecyclelr':
        return OneCycleLR(optimizer, **kwargs)
    elif scheduler.lower() == 'cosineannealingwarmrestarts':
        return CosineAnnealingWarmRestarts(optimizer, **kwargs)
    else:
        # To use custom scheduler e.g. your_module.some_file:ClassName
        scheduler_class = dynamic_import(scheduler)
        return scheduler_class(optimizer, **kwargs)
