from abc import ABC
from abc import abstractmethod
from distutils.version import LooseVersion

import torch
import torch.optim.lr_scheduler as L


# FIXME(kamo): EpochScheduler and BatchScheduler are confusing names.


# If you need to define custom scheduler, please inherit these classes
class AbsBatchScheduler(ABC):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


class AbsEpochScheduler(ABC):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


class AbsValEpochScheduler(AbsBatchScheduler):
    @abstractmethod
    def step(self, val, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


# Create alias type to check the type
# Note(kamo): Currently PyTorch doesn't provide the base class
# to judge these classes.
AbsValEpochScheduler.register(L.ReduceLROnPlateau)
AbsEpochScheduler.register(L.ReduceLROnPlateau)
AbsEpochScheduler.register(L.LambdaLR)
AbsEpochScheduler.register(L.StepLR)
AbsEpochScheduler.register(L.MultiStepLR)
AbsEpochScheduler.register(L.ExponentialLR)
AbsEpochScheduler.register(L.CosineAnnealingLR)

if LooseVersion(torch.__version__) >= LooseVersion('1.1.0'):
    AbsBatchScheduler.register(L.CyclicLR)
    AbsBatchScheduler.register(L.OneCycleLR)
    AbsBatchScheduler.register(L.CosineAnnealingWarmRestarts)
