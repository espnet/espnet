from abc import ABC

# from torch.optim.lr_scheduler import CyclicLR
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typeguard import typechecked, check_argument_types
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from typing import Type

from espnet.utils.dynamic_import import dynamic_import


# FIXME(kamo): EpochScheduler and BatchScheduler are confusing. Please give me an idea.


# If you need to define custom scheduler, please inherit these classes
class AbsBatchScheduler(ABC):
    def step(self, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass


class AbsEpochScheduler(ABC):
    def step(self, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass


class AbsValEpochScheduler(AbsBatchScheduler):
    def step(self, val, epoch: int = None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
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
def get_epoch_scheduler_class(scheduler: str) -> Type[AbsEpochScheduler]:
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
    if scheduler.lower() == 'reducelronplateau':
        return ReduceLROnPlateau
    elif scheduler.lower() == 'lambdalr':
        return LambdaLR
    elif scheduler.lower() == 'steplr':
        return StepLR
    elif scheduler.lower() == 'multisteplr':
        return MultiStepLR
    elif scheduler.lower() == 'exponentiallr':
        return ExponentialLR
    elif scheduler.lower() == 'cosineannealinglr':
        return CosineAnnealingLR
    else:
        # To use custom scheduler e.g. your_module.some_file:ClassName
        scheduler_class = dynamic_import(scheduler)
        return scheduler_class


@typechecked
def get_batch_scheduler_class(scheduler: str) -> Type[AbsBatchScheduler]:
    """PyTorch schedulers control optim-parameters after every updating

    BatchScheduler:
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """
    # Note(kamo): Don't use getattr or dynamic_import
    # for readability and debuggability as possible
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
