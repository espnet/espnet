#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Abstract base class for job templates in SpeechLM training."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch.nn as nn


class AbsJobTemplate(ABC):
    """Abstract base class for training job templates.

    Defines the data flow from raw sample dictionaries to model loss computation.
    Subclasses implement two key methods: build_preprocessor() and build_model().

    This abstraction enables support for diverse training paradigms including
    SpeechLM, diffusion models, and self-supervised learning by customizing
    the preprocessing pipeline and model architecture.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the job template with configuration.

        Args:
            config: Job configuration containing model architecture, data
                processing parameters, and training settings.
        """
        self.config = config

    @abstractmethod
    def build_preprocessor(self) -> Callable:
        """Build and return the preprocessor object.

        The preprocessor object should implement three key methods:

        1. preprocessing(data_dict: Dict[str, Any]) -> Dict[str, Any]:
           Converts a single raw data dictionary into a training-ready example.
           This method handles per-sample transformations and can be used
           both during training (within collate_fn) and inference stages.

        2. collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
           Combines multiple training-ready samples into a batch for training.
           This method typically calls preprocessing() on each sample first,
           then batches the results together. The collate_fn is passed to
           PyTorch DataLoader for multi-processing data loading.

        3. find_length(data_dict: Dict[str, Any]) -> int:
           Quickly computes the sample length from a raw data dictionary.
           Used for efficient batch construction and data statistics collection
           without performing full preprocessing.

        Note: When using PyTorch DataLoader with num_workers > 0, the collate_fn
        executes in worker subprocesses. Ensure the preprocessor object is
        picklable and avoid CUDA operations as workers don't have GPU access.

        Returns:
            A preprocessor object with preprocessing(), collate_fn(), and
            find_length() methods.
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model for training.

        Constructs the model architecture based on the configuration
        provided during initialization. The model should implement
        forward() to compute loss and return training statistics.

        Returns:
            PyTorch model instance ready for training with DeepSpeed.
        """
        raise NotImplementedError
