"""VoxCeleb speaker verification dataset module."""

from egs3.voxceleb.esp2_spk.dataset.builder import VoxCelebBuilder as DatasetBuilder
from egs3.voxceleb.esp2_spk.dataset.dataset import VoxCelebDataset as Dataset

# Imported here, before Lightning builds the dataloaders, so that Lightning
# records its arguments and can rebuild it around a DistributedSampler.
from egs3.voxceleb.esp2_spk.dataset.sampler import SpeakerBalancedBatchSampler

__all__ = ["Dataset", "DatasetBuilder", "SpeakerBalancedBatchSampler"]
