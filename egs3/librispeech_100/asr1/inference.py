from omegaconf import OmegaConf
from pathlib import Path
from hydra.utils import instantiate
import torch.nn as nn

from espnet3.data import DataOrganizer
from espnet3.inference_runner import InferenceRunner


import time
import numpy as np
from espnet2.bin.asr_inference_ctc import Speech2Text


class ASRInferenceWrapper(nn.Module):
    def __init__(self, config_path: str, model_path: str):
        super().__init__()
        self.model = Speech2Text(
            config_path, model_path
        )

    def __call__(self, sample: dict) -> dict:
        assert "speech" in sample, "Missing 'speech' key in sample"
        speech = sample["speech"]

        start = time.time()
        results = self.model(speech)
        end = time.time()

        hyp_text = results[0][0] if results else ""
        duration = len(speech) / 16000  # 16kHz assumption
        elapsed = end - start
        rtf = elapsed / duration if duration > 0 else 0.0

        output = {
            "hypothesis": {"type": "text", "value": hyp_text},
            "rtf": {"type": "text", "value": str(round(rtf, 4))},
        }

        # Add reference text if available
        if "text" in sample:
            text = self.model.tokenizer.tokens2text(
                self.model.converter.ids2tokens(sample["text"])
            )
            output["text"] = {"type": "text", "value": text}

        return output


def main():
    config_path = "egs3/librispeech_100/asr1/config.yaml"
    inference_config = "egs3/librispeech_100/asr1/inference.yaml"
    # model_config = "egs3/librispeech_100/asr1/exp/gpu1/config.yaml"
    # model_ckpt = "egs3/librispeech_100/asr1/exp/gpu1/epoch46_step67069_valid.loss.ckpt"

    # load full experiment config
    train_config = OmegaConf.load(config_path)
    inference_config = OmegaConf.load(inference_config)

    # run inference
    runner = InferenceRunner(
        config=train_config,
        model_config=model_config,
        decode_dir=Path(train_config.expdir) / "decode_parallel",
        parallel=train_config.parallel,
        resume=True,
    )

    runner.run()
    runner.compute_metrics(train_config.test)


if __name__ == "__main__":
    main()


