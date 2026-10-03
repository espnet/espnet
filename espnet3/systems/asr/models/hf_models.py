"""Hugging Face model wrappers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import lightning
import torch
from transformers import AutoModel, AutoProcessor

from espnet2.torch_utils.device_funcs import force_gatherable


class AbsHFTrainingWrapper(lightning.LightningModule, ABC):
    """Common interface for training Hugging Face models in ESPnet.

    This class serves as an abstract superclass that allows Hugging Face
    models to be fine-tuned in ESPnet. It does so by acting as a wrapper
    around models from the `transformers` package.

    To train a Hugging Face model in ESPnet, create a subclass of
    AbsHFTrainingWrapper and implement/overwrite the methods below.
    Getting the training data into the format expected by the model may
    also require writing a custom collate function.
    See ``egs3/librispeech_100/asr/src/granite_speech.py`` for examples.

    While many elements of the transformers package are standardized, there are still
    differences between models when it comes to naming conventions, preprocessing, etc.
    This makes it difficult to provide a universal wrapper that works with all
    Hugging Face models. As such, this class provides default implementations that can
    be overwritten when inheriting from it if necessary.

    In order to be able to load the fine-tuned model during the inference
    stage, it must be saved using the ``HFCheckpointSaveCallback``.
    See ``espnet3.components.callbacks.HFCheckpointSaveCallback`` for more
    information.

    For more information on how to load and train a model, refer to the its
    page on Hugging Face.
    """

    model_class = AutoModel
    processor_class = AutoProcessor

    def __init__(self, model_tag_or_path: str, **kwargs):
        """Load the model and processor and performs any additional setup.

        The model and processor class are defined using the model_class and
        processor_class attributes.
        This means that subclasses only need to define __init__() if additional
        setup is necessary.

        This class can be instantiated by putting the following in the YAML
        config (replace _target_ with a subclass for a specific model):
        ```yaml
        model:
          _target_: espnet3.systems.asr.models.AbsHFTrainingWrapper
          model_tag_or_path: {Hugging Face tag or path to a local model}
        ```

        Args:
            model_tag_or_path (str): Hugging Face model tag or path to a local model.
        """
        super().__init__()
        self.model = self.model_class.from_pretrained(model_tag_or_path)
        self.processor = self.processor_class.from_pretrained(model_tag_or_path)

    def forward(
        self, **batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Perform a forward pass and return the loss, stats, and batch weights.

        Args:
            batch: Batched output of the collate function.

        Returns:
            3-tuple of (loss, stats, weight).
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        stats = {"loss": loss.detach()}
        batch_size = outputs.logits.shape[0]

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @abstractmethod
    def collect_feats(self, **batch) -> Dict[str, torch.Tensor]:
        """Return the input features and (if applicable) corresponding lengths.

        Example implementation for IBM Granite:
        ```python
        def collect_feats(self, **batch) -> Dict[str, torch.Tensor]:
            feats = batch["input_features"]
            feats_lengths = batch["input_features_mask"].sum(dim=-1)
            return {"feats": feats, "feats_lengths": feats_lengths}
        ```

        Args:
            batch: Batched output of the collate function.

        Returns:
            Dict of features and feature lengths.
        """
        raise NotImplementedError

    def save_pretrained(self, dirpath):
        """Save the model and processor together to dirpath.

        This method generally shouldn't be overwritten unless custom saving logic
        is needed.

        Args:
            dirpath: Directory or path where the model and processor are saved.
        """
        self.model.save_pretrained(dirpath)
        self.processor.save_pretrained(dirpath)


class AbsHFInferenceWrapper(lightning.LightningModule, ABC):
    """Common interface for performing inference using Hugging Face models in ESPnet.

    This class serves as an abstract superclass that allows Hugging Face
    models to be used for running inference in ESPnet. It does so by acting
    as a wrapper around models from the `transformers` package.

    To run inference using a Hugging Face model in ESPnet, create a subclass of
    AbsHFInferenceWrapper and implement/overwrite the methods below.
    See ``egs3/librispeech_100/asr/src/granite_speech.py`` for an example.

    While many elements of the transformers package are standardized,
    there are still differences between models when it comes to naming conventions,
    preprocessing steps, etc. This makes it difficult to provide a universal wrapper
    that works with all Hugging Face models.
    As such, this class provides default implementations that can be overwritten when
    inheriting from it if necessary.

    In order to load a model fine-tuned in ESPnet for inference, be sure to
    use the ``HFCheckpointSaveCallback`` during training to save the model
    in Hugging Face's format.

    For more information on how to load and perform inference, refer to the model's
    page on Hugging Face.
    """

    model_class = AutoModel
    processor_class = AutoProcessor

    def __init__(self, model_tag_or_path: str, **kwargs):
        """Load the model and processor and performs any additional setup.

        The model and processor class are defined using the model_class and
        processor_class attributes. This means that subclasses only need to define
        __init__() if additional setup is necessary.


        To load the best fine-tuned model saved with
        ``HFCheckpointSaveCallback`` (replace _target_ with a subclass for
        a specific model):
        ```yaml
        model:
          _target_: espnet3.systems.asr.models.AbsHFInferenceWrapper
          model_tag_or_path: ${exp_dir}/hf
        ```

        Args:
            model_tag_or_path (str): Hugging Face model tag or path to a local model.
        """
        super().__init__()
        self.model = self.model_class.from_pretrained(model_tag_or_path)
        self.processor = self.processor_class.from_pretrained(model_tag_or_path)

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Perform inference and returns the outputs.

        For ASR, the input to this function is typically a tensor of
        the raw input audio. Details on the matter (e.g. batch support)
        can be found on the model's page on Hugging Face.

        Example implementation for IBM Granite:
        ```python
        def forward(self, speech: torch.Tensor):
            inputs = self.processor(
                text=self.prompt,
                audio=speech,
                return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
            )

            num_input_tokens = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
            output_text = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )

            return [output_text]
        ```

        Args:
            inputs: Inputs to the model.

        Returns:
            Inference results.
        """
        raise NotImplementedError
