from abc import ABC

import torch


# FIXME(kamo): EpochScheduler and BatchScheduler are confusing names.


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
AbsValEpochScheduler.register(torch.optim.lr_scheduler.ReduceLROnPlateau)
AbsEpochScheduler.register(torch.optim.lr_scheduler.ReduceLROnPlateau)
AbsEpochScheduler.register(torch.optim.lr_scheduler.LambdaLR)
AbsEpochScheduler.register(torch.optim.lr_scheduler.StepLR)
AbsEpochScheduler.register(torch.optim.lr_scheduler.MultiStepLR)
AbsEpochScheduler.register(torch.optim.lr_scheduler.ExponentialLR)
AbsEpochScheduler.register(torch.optim.lr_scheduler.CosineAnnealingLR)

try:
    AbsBatchScheduler.register(torch.optim.lr_scheduler.CyclicLR)
    AbsBatchScheduler.register(torch.optim.lr_scheduler.OneCycleLR)
    AbsBatchScheduler.register(
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
except AttributeError:
    pass
