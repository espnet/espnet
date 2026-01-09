"""ASR inference provider and runner implementations.

This module wires dataset/model construction and per-sample inference for
automatic speech recognition (ASR) evaluation and decoding.
"""

import logging
import os

import torch
from hydra.utils import instantiate

from espnet3.systems.base.inference_provider import (
    InferenceProvider as BaseInferenceProvider,
)
from espnet3.systems.base.inference_runner import AbsInferenceRunner

logger = logging.getLogger(__name__)


class InferenceProvider(BaseInferenceProvider):
    """Inference provider for ASR datasets and models.

    This provider builds the test dataset and model instance using the
    configuration supplied by Hydra/OmegaConf.
    """

    @staticmethod
    def build_dataset(config):
        """Build the test dataset from the provided config.

        Args:
            config: Configuration with dataset definition and test_set name.

        Returns:
            Dataset object for the selected test split.

        Raises:
            AttributeError: If required dataset config fields are missing.
        """
        # config includes test dataset
        organizer = instantiate(config.dataset)
        test_set = config.test_set
        logger.info("Building dataset for test set: %s", test_set)
        return organizer.test[test_set]

    @staticmethod
    def build_model(config):
        """Instantiate the ASR model on the selected device.

        Args:
            config: Configuration with model definition.

        Returns:
            Instantiated model moved to the selected device.

        Raises:
            Exception: If model instantiation fails.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            device = f"cuda:{device_id}"

        # config includes model
        logger.info(
            "Instantiating model %s on %s",
            getattr(config.model, "_target_", None),
            device,
        )
        model = instantiate(
            config.model, device=device
        )  # In this recipe we assume this to be espnet2.bin.asr_inference.Speech2Text
        return model


class InferenceRunner(AbsInferenceRunner):
    """Inference runner that produces hypotheses and references.

    This runner expects dataset items to contain ``speech`` and ``text``
    fields and returns decoded hypotheses alongside references.

    It is designed around ``espnet3.systems.asr.task.ASRTask`` outputs; when
    using a custom model outside that task, implement your own inference
    runner and decoding logic.
    """

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Run inference for one dataset item and return hypothesis/ref.

        Args:
            idx: Integer index into the dataset.
            dataset: Dataset providing speech/text entries.
            model: ASR inference model callable on speech input.
            **kwargs: Unused keyword arguments reserved for compatibility.

        Returns:
            Dict with index, hypothesis string, and reference string.

        Raises:
            AssertionError: If required fields are missing from dataset items.
        """
        data = dataset[idx]
        assert "speech" in data, "ASR inference requires 'speech' in dataset item."
        assert "text" in data, "ASR inference requires 'text' in dataset item."
        speech = data["speech"]
        hyp = model(speech)[0][0]
        ref = data["text"]
        return {"idx": idx, "hyp": hyp, "ref": ref}



class TransducerInferenceRunner(AbsInferenceRunner):
    """Inference runner that produces hypotheses and references.

    This runner expects dataset items to contain ``speech`` and ``text``
    fields and returns decoded hypotheses alongside references.

    It is designed around ``espnet3.systems.asr.task.ASRTask`` outputs; when
    using a custom model outside that task, implement your own inference
    runner and decoding logic.
    """

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Run inference for one dataset item and return hypothesis/ref.

        Args:
            idx: Integer index into the dataset.
            dataset: Dataset providing speech/text entries.
            model: ASR inference model callable on speech input.
            **kwargs: Unused keyword arguments reserved for compatibility.

        Returns:
            Dict with index, hypothesis string, and reference string.

        Raises:
            AssertionError: If required fields are missing from dataset items.
        """
        data = dataset[idx]
        assert "speech" in data, "ASR inference requires 'speech' in dataset item."
        assert "text" in data, "ASR inference requires 'text' in dataset item."
        speech = data["speech"]
        hyp = model(speech)[0]
        ref = data["text"]
        return {"idx": idx, "hyp": hyp, "ref": ref}
