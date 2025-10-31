import argparse
import os
from pathlib import Path

import torch
from hydra.utils import instantiate

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.inference_provider import InferenceProvider
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.config import load_config_with_defaults


class DecodeProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        # config includes test dataset
        organizer = instantiate(config.dataset)
        test_set = config.test_set
        return organizer.test[test_set]

    @staticmethod
    def build_model(config):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            device = f"cuda:{device_id}"

        # config includes model
        model = instantiate(
            config.model, device=device
        )  # In this recipe we assume this to be espnet2.bin.asr_inference.Speech2Text
        return model


class DecodeRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        speech = data["speech"]
        hyp = model(speech)[0][0]
        ref = model.tokenizer.tokens2text(model.converter.ids2tokens(data["text"]))
        return {"idx": idx, "hyp": hyp, "ref": ref}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_config.yaml")
    args = parser.parse_args()

    config = load_config_with_defaults(args.config)
    set_parallel(config.parallel)

    test_sets = [test_set.name for test_set in config.dataset.test]
    assert len(test_sets) > 0, "No test set found in dataset"
    assert len(test_sets) == len(set(test_sets)), f"Duplicate test key found."

    for test_name in test_sets:
        print(f"===> Processing {test_name}")
        config.test_set = test_name
        provider = DecodeProvider(config)
        runner = DecodeRunner(
            provider=provider,
            async_mode=False,
        )
        dataset_length = len(provider.build_dataset(config))
        print(f"===> Processing {dataset_length} samples..")
        out = runner(list(range(dataset_length)))

        # create scp files
        (Path(config.decode_dir) / test_name).mkdir(parents=True, exist_ok=True)
        with open(Path(config.decode_dir) / test_name / "ref.scp", "w") as f:
            f.write("\n".join([f"{result['idx']} {result['ref']}" for result in out]))

        with open(Path(config.decode_dir) / test_name / "hyp.scp", "w") as f:
            f.write("\n".join([f"{result['idx']} {result['hyp']}" for result in out]))
