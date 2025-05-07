import argparse
import time
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf

# from espnet2.bin.asr_inference_ctc import Speech2Text
from espnet2.bin.s2t_inference import Speech2Text
from espnet3.inference.inference_runner import InferenceRunner


class ASRInferenceWrapper(nn.Module):
    def __init__(self, config_path: str, model_path: str):
        super().__init__()
        self.model = Speech2Text(config_path, model_path)

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
            "elapsed_time": {"type": "text", "value": str(round(elapsed, 4))},
        }

        if "text" in sample:
            text = self.model.tokenizer.tokens2text(
                self.model.converter.ids2tokens(sample["text"])
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

    runner = InferenceRunner(
        config=train_config,
        model_config=inference_config,
        decode_dir=decode_dir,
        parallel=train_config.parallel,
        resume=not args.no_resume,
    )

    runner.run()
    runner.compute_metrics(train_config.test)


if __name__ == "__main__":
    main()
