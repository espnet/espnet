from espnet2.torch_utils.device_funcs import force_gatherable

from transformers import AutoModel, AutoProcessor
import torch
import lightning
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod


class AbsHFTrainingWrapper(lightning.LightningModule, ABC):
    """
    This class provides a common interface for training Hugging Face models in ESPnet.
    While many elements of the transformers package are standardized, there are still
    differences between models when it comes to naming conventions, preprocessing steps, etc.
    This makes it difficult to provide a universal wrapper that works with all Hugging Face models.
    As such, this class provides default implementations that can be overwritten when
    inheriting from it if necessary.
    """
    model_class = AutoModel
    processor_class = AutoProcessor

    def __init__(self, model_tag_or_path: str, **kwargs):
        """
        Loads the model and processor and performs any additional setup.
        The model and processor class are defined using the model_class and processor_class attributes.
        This means that subclasses only need to define __init__() if additional setup is necessary.

        Args:
            model_tag_or_path (str): Hugging Face model tag or path to a local model.
        """
        super().__init__()
        self.model = self.model_class.from_pretrained(model_tag_or_path)
        self.processor = self.processor_class.from_pretrained(model_tag_or_path)

    def forward(
        self,
        **batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a forward pass and returns the loss, stats, and batch weights.

        Args:
            batch: Batched output of the collate function.

        Returns:
            3-tuple of (loss, stats, weight).
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        stats = {"loss": loss}
        batch_size = outputs.logits.shape[0]

        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @abstractmethod
    def collect_feats(self, **batch) -> Dict[str, torch.Tensor]:
        """
        Returns the input features and (if applicable) corresponding lengths for the input batch.

        Args:
            batch: Batched output of the collate function.

        Returns:
            Dict of features and feature lengths.
        """
        raise NotImplementedError

    def save_pretrained(self, dirpath):
        """
        Saves the model and processor together to dirpath.
        This method generally shouldn't be overwritten unless custom saving logic is needed.

        Args:
            dirpath: Directory or path where the model and processor ared saved.
        """
        self.model.save_pretrained(dirpath)
        self.processor.save_pretrained(dirpath)


class AbsHFInferenceWrapper(lightning.LightningModule, ABC):
    """
    This class provides a common interface for performing inference using Hugging Face models in ESPnet.
    While many elements of the transformers package are standardized, there are still
    differences between models when it comes to naming conventions, preprocessing steps, etc.
    This makes it difficult to provide a universal wrapper that works with all Hugging Face models.
    As such, this class provides default implementations that can be overwritten when
    inheriting from it if necessary.

    For more information on how to load and perform inference, refer to the model's page
    on Hugging Face.
    """
    model_class = AutoModel
    processor_class = AutoProcessor
    
    def __init__(self, model_tag_or_path: str, **kwargs):
        """
        Loads the model and processor and performs any additional setup.
        The model and processor class are defined using the model_class and processor_class attributes.
        This means that subclasses only need to define __init__() if additional setup is necessary.

        Args:
            model_tag_or_path (str): Hugging Face model tag or path to a local model.
        """
        super().__init__()
        self.model = self.model_class.from_pretrained(model_tag_or_path)
        self.processor = self.processor_class.from_pretrained(model_tag_or_path)

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """
        Performs inference and returns the outputs.

        Args:
            inputs: Inputs to the model.

        Returns:
            Inference results.
        """
        raise NotImplementedError
