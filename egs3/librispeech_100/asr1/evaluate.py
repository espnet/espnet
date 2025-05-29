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
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        InferenceRunner.__init__(self, **kwargs)

    def initialize_model(self, device=None):
        if device is None:
            device = self.device
        return instantiate(self.model_config, device=device)

    def inference_body(self, model, sample: dict) -> dict:
        assert "speech" in sample, "Missing 'speech' key in sample"
        speech = sample["speech"]

        start = time.time()
        results = model(speech)
        end = time.time()

        hyp_text = results[0][0] if results else ""
        duration = len(speech) / 16000
        elapsed = end - start
        rtf = elapsed / duration if duration > 0 else 0.0
        output = {
            "hypothesis": {"type": "text", "value": hyp_text},
            "rtf": {"type": "text", "value": str(round(rtf, 4))},
        }
        if "text" in sample:
            text = model.tokenizer.tokens2text(
                model.converter.ids2tokens(sample["text"])
            )
            output["ref"] = {"type": "text", "value": text}

        return output


def run_single_sample_inference(config_path):
    config = OmegaConf.load(config_path)

    runner = ASRInferenceRunner(
        model_config=config.model,
        dataset_config=config.dataset,
    )
    test_key = config.dataset.test[0].name
    print(test_key, "TEST_KEY")
    dataset = runner.initialize_dataset(test_key)
    uid, sample = dataset[0]

    result = runner.run_on_example(uid, sample)

    print("==== DEBUG INFERENCE RESULT ====")
    for key, val in result.items():
        print(f"{key}: {val['value']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="evaluate.yaml",
        help="Path to evaluation config (e.g., evaluate.yaml)",
    )
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode")
    parser.add_argument("--debug_sample", action="store_true",
                        help="Run debug inference on one sample")

    args = parser.parse_args()

    if args.debug_sample:
        run_single_sample_inference(args.config)
    else:
        config = OmegaConf.load(args.config)

        runner = ASRInferenceRunner(
            model_config=config.model,
            dataset_config=config.dataset,
            parallel=config.parallel,
        )
        test_keys = [ds_conf.name for ds_conf in config.dataset.test]

        for test_key in test_keys:
            runner.run_on_dataset(test_key, output_dir=f"{config.decode_dir}/{test_key}")

        # runner.compute_metrics(train_config.test)


if __name__ == "__main__":
    main()
