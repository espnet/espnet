"""Opt-in strict-load and fine-tuning check for downloaded ESPnet model packages.

Run from the repository root with PYTHONPATH=. Model package files must already
be available; the frontend library may download weights if its cache is incomplete.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from espnet2.tasks.aqa import AqaTask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frontend-cache", type=Path)
    parser.add_argument("--metrics", required=True, type=json.loads)
    options = parser.parse_args(argv)
    if not isinstance(options.metrics, dict) or not options.metrics:
        parser.error("--metrics must be a non-empty JSON object")
    torch.manual_seed(0)
    config = yaml.safe_load((options.package / options.config).read_text())
    for name in (
        "metric2id",
        "metric2type",
        "metric_token_info",
        "bpemodel",
        "token_list",
    ):
        if isinstance(config.get(name), str):
            path = options.package / config[name]
            if not path.is_file():
                raise FileNotFoundError(f"Missing published {name}: {path}")
            config[name] = str(path)
    if options.frontend_cache:
        config["frontend_conf"]["download_dir"] = str(options.frontend_cache)
    args = argparse.Namespace(**config)
    model = AqaTask.build_model(args)
    unknown = set(options.metrics) - set(model.universa.metric2id)
    if unknown:
        raise ValueError(f"Metrics absent from this checkpoint: {sorted(unknown)}")
    for name, value in options.metrics.items():
        categorical = (args.metric2type or {}).get(name) == "categorical"
        if not categorical and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value == args.metric_pad_value
        ):
            raise ValueError(f"Expected a finite, unmasked numeric label for {name}")
    state = torch.load(
        options.package / options.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    print("STRICT_LOAD", model.load_state_dict(state, strict=True), flush=True)
    del state
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in config.get("freeze_param", [])):
            parameter.requires_grad_(False)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    audio = torch.randn(1, 16000) * 0.05
    lengths = torch.tensor([audio.size(1)])
    references = {}
    if model.use_ref_audio:
        references.update(ref_audio=audio, ref_audio_lengths=lengths)
    if model.use_ref_text:
        references.update(
            ref_text=torch.tensor([[1]]), ref_text_lengths=torch.tensor([1])
        )
    if getattr(model.universa, "sequential_metrics", False):
        pairs = model.universa.metric_tokenizer.metric2token(options.metrics)
        tokens = torch.tensor([[token for pair in pairs.values() for token in pair]])
        labels = {
            "metric_token": tokens,
            "metric_token_lengths": torch.tensor([tokens.size(1)]),
        }
    else:
        labels = {key: torch.tensor([value]) for key, value in options.metrics.items()}
    loss, _, _ = model(audio, lengths, labels, **references)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    updated = next(
        p for p in parameters if p.grad is not None and p.grad.abs().sum() > 0
    )
    before = updated.detach().clone()
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, updated)
    print("FINETUNE_STEP", loss.item(), flush=True)
    model.eval()
    if getattr(model.universa, "sequential_metrics", False):
        model.universa.set_inference(1, list(options.metrics), True)
    with torch.no_grad():
        predictions = model.inference(audio, lengths, **references)
    for name in options.metrics:
        values = predictions.get(name)
        if values is None or len(values) != 1:
            raise ValueError(f"Expected one prediction for {name}")
        if (args.metric2type or {}).get(name) == "categorical":
            if values[0] not in model.universa.metric_tokenizer.tokenizer_config[name]:
                raise ValueError(f"Invalid categorical prediction for {name}")
        elif not np.isfinite(values).all():
            raise ValueError(f"Non-finite prediction for {name}")
    print("PREDICTED_METRICS", sorted(options.metrics), flush=True)


if __name__ == "__main__":
    main()
