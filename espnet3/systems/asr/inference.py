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
    @staticmethod
    def build_dataset(config):
        # config includes test dataset
        organizer = instantiate(config.dataset)
        test_set = config.test_set
        logger.info("Building dataset for test set: %s", test_set)
        return organizer.test[test_set]

    @staticmethod
    def build_model(config):
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
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        assert "speech" in data, "ASR inference requires 'speech' in dataset item."
        assert "text" in data, "ASR inference requires 'text' in dataset item."
        speech = data["speech"]
        hyp = model(speech)[0][0]
        ref = data["text"]
        return {"idx": idx, "hyp": hyp, "ref": ref}
