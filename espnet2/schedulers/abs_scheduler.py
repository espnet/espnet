from abc import ABC
from abc import abstractmethod
from distutils.version import LooseVersion

import torch
import torch.optim.lr_scheduler as L


class AbsScheduler(ABC):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


# If you need to define custom scheduler, please inherit these classes
class AbsBatchStepScheduler(AbsScheduler):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


class AbsEpochStepScheduler(AbsScheduler):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


class AbsValEpochStepScheduler(AbsEpochStepScheduler):
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
AbsValEpochStepScheduler.register(L.ReduceLROnPlateau)
for s in [
    L.ReduceLROnPlateau,
    L.LambdaLR,
    L.StepLR,
    L.MultiStepLR,
    L.MultiStepLR,
    L.ExponentialLR,
    L.CosineAnnealingLR,
]:
    AbsEpochStepScheduler.register(s)
if LooseVersion(torch.__version__) >= LooseVersion("1.3.0"):
    for s in [L.CyclicLR, L.OneCycleLR, L.CosineAnnealingWarmRestarts]:
        AbsBatchStepScheduler.register(s)
