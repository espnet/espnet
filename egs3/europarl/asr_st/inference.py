import argparse
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from espnet2.bin.s2t_inference import Speech2Text
from espnet3.inference.inference_runner import InferenceRunner


class OWSMInferenceRunner(InferenceRunner, nn.Module):
    def __init__(self, ckpt_path, model_tag, **kwargs):
        nn.Module.__init__(self)
        InferenceRunner.__init__(self, **kwargs)
        self.ckpt_path = ckpt_path
        self.model_tag = model_tag

    def initialize_model(self, device=None):
        if device is None:
            device = self.device
        model = Speech2Text.from_pretrained(self.model_tag, device=device)
        d = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        # remove the initial "model."
        state_dict = {k[6:]: v for k, v in d["state_dict"].items()}
        model.s2t_model.load_state_dict(state_dict)
        return model

    def inference_body(self, model, sample: dict) -> dict:
        assert "speech" in sample, "Missing 'speech' key in sample"
        speech = sample["speech"]

        lang_id = sample["text"][1]
        task_id = sample["text"][2]

        lang_sym = model.converter.ids2tokens([lang_id])[0]
        task_sym = model.converter.ids2tokens([task_id])[0]

        start = time.time()
        results = model(speech, lang_sym=lang_sym, task_sym=task_sym)
        end = time.time()

        hyp_text = results[0][3] if results else ""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config (e.g., config.yaml)",
    )
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    runner = OWSMInferenceRunner(
        model_config=config.model,
        dataset_config=config.dataset,
        parallel=config.parallel,
        ckpt_path=config.ckpt_path,
        model_tag=config.model.model_tag,
    )
    test_keys = [ds_conf.name for ds_conf in config.dataset.test]

    for test_key in test_keys:
        runner.run_on_dataset(test_key, output_dir=f"{config.decode_dir}/{test_key}")

    # runner.compute_metrics(train_config.test)


if __name__ == "__main__":
    main()
