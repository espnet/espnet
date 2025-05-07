import argparse
import time
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

# from espnet2.bin.asr_inference_ctc import Speech2Text
from espnet2.bin.asr_inference import Speech2Text
from espnet3.inference.inference_runner import InferenceRunner


class ASRInferenceRunner(InferenceRunner, nn.Module):
    def __init__(self, config_path: str, model_path: str):
        InferenceRunner.__init__(self)
        nn.Module.__init__(self)
        self.config_path = config_path
        self.model_path = model_path
        self.asr_model = None

    def initialize(self, sample: dict):
        if self.asr_model is None:
            self.asr_model = Speech2Text(self.config_path, self.model_path, device="cuda")
            
    def inference_body(self, sample: dict) -> dict:
        assert "speech" in sample, "Missing 'speech' key in sample"
        speech = sample["speech"]
        start = time.time()
        results = self.asr_model(speech)
        end = time.time()
        hyp_text = results[0][0] if results else ""
        duration = len(speech) / 16000  # assuming 16kHz
        elapsed = end - start
        rtf = elapsed / duration if duration > 0 else 0.0
        output = {
            "hypothesis": {"type": "text", "value": hyp_text},
            "rtf": {"type": "text", "value": str(round(rtf, 4))},
        }
        if "text" in sample:
            text = self.asr_model.tokenizer.tokens2text(
                self.asr_model.converter.ids2tokens(sample["text"])
            )
            output["text"] = {"type": "text", "value": text}

        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to training config (e.g., config.yaml)",
    )
    parser.add_argument(
        "--inference_config",
        type=str,
        required=True,
        help="Path to inference config (e.g., inference.yaml)",
    )
    parser.add_argument(
        "--decode_dir", type=str, default=None, help="Directory to save decode results"
    )
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode")

    args = parser.parse_args()

    train_config = OmegaConf.load(args.config_path)
    inference_config = OmegaConf.load(args.inference_config)

    decode_dir = (
        Path(args.decode_dir)
        if args.decode_dir
        else Path(train_config.expdir) / "decode"
    )

    runner = ASRInferenceRunner(
        config=train_config,
        model_config=inference_config,
        decode_dir=decode_dir,
        # parallel=train_config.parallel,
        # resume=not args.no_resume,
    )

    ds = instantiate(train_config.dataset)

    for test_keys in ds.test.keys():
        runner.run_on_dataset(
            ds.test[test_keys],
            output_dir=f"decode/{test_keys}"
        )

    # runner.compute_metrics(train_config.test)


if __name__ == "__main__":
    main()

