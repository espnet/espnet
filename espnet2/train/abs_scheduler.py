from abc import ABC

# from torch.optim.lr_scheduler import CyclicLR
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


# FIXME(kamo): EpochScheduler and BatchScheduler are confusing names. Please give naming.


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
