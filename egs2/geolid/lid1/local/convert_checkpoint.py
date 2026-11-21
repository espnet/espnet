#!/usr/bin/env python3
# Copyright 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Convert original GeoLID weights to the stock-S3PRL layer-wrapper layout.

No parameters are retrained or numerically modified. Runtime uses current ESPnet
pooling and normalization; historical metrics must not be reported as a new
benchmark of the converted model.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import yaml

from espnet2.tasks.lid import LIDTask
from espnet2.train.trainer import TrainerOptions

PREFIX = "frontend.upstream.upstream.model.encoder."


def convert_state_dict(source, layers):
    """Rename weights, checking redundant aliases before discarding them."""
    canonical = {}
    for name, value in source.items():
        if name.startswith("upstream_encoder."):
            name = PREFIX + name.removeprefix("upstream_encoder.")
        if name.startswith(PREFIX + "aamsoftmax_loss."):
            name = "loss." + name.removeprefix(PREFIX + "aamsoftmax_loss.")
        if name.startswith("loss.lang2vec_head.0."):
            name = "lang2vec_head." + name.removeprefix("loss.lang2vec_head.0.")
        if name in canonical and not torch.equal(canonical[name], value):
            raise ValueError(f"Conflicting checkpoint aliases: {name}")
        canonical[name] = value

    converted = {}
    for name, value in canonical.items():
        targets = [name]
        if name.startswith(PREFIX):
            suffix = name.removeprefix(PREFIX)
            parts = suffix.split(".")
            group = parts[0]
            if (
                group == "layers"
                and int(parts[1]) in layers
                and parts[2]
                not in {
                    "layer",
                    "encoder",
                    "pooling",
                    "projector",
                    "prediction",
                    "conditioning_layer",
                }
            ):
                targets = [PREFIX + f"layers.{parts[1]}.layer." + ".".join(parts[2:])]
            elif group in {"ecapa_encoder", "pooling", "projector", "lang2vec_head"}:
                module = {
                    "ecapa_encoder": "encoder",
                    "lang2vec_head": "prediction",
                }.get(group, group)
                parameter = ".".join(
                    parts[3:] if group == "lang2vec_head" else parts[2:]
                )
                targets = [PREFIX + f"layers.{parts[1]}.{module}.{parameter}"]
            elif group in {"conditioning_projs", "lang2vec_conditioning_projs"}:
                if len(parts) == 2:
                    targets = [
                        PREFIX + f"layers.{i}.conditioning_layer.{parts[1]}"
                        for i in layers
                    ]
                else:
                    parameter = parts[-1]
                    targets = [
                        PREFIX + f"layers.{parts[1]}.conditioning_layer.{parameter}"
                    ]
        for target in targets:
            if target in converted and not torch.equal(converted[target], value):
                raise ValueError(f"Conflicting converted weights: {target}")
            converted[target] = value
    return converted


def convert_config(source):
    """Keep architecture settings and use the stock ESPnet inference path."""
    modules = (
        "model",
        "frontend",
        "specaug",
        "normalize",
        "encoder",
        "pooling",
        "projector",
        "encoder_condition",
        "pooling_condition",
        "projector_condition",
        "loss",
    )
    keys = {key for module in modules for key in (module, module + "_conf")}
    config = {key: value for key, value in source.items() if key in keys}
    # lid_inference builds TrainerOptions from the saved training configuration.
    # Use current defaults for fields absent from older training runs.
    defaults = vars(LIDTask.get_parser().parse_args([]))
    for field in dataclasses.fields(TrainerOptions):
        config[field.name] = source.get(field.name, defaults[field.name])
    config["model_conf"] = dict(config["model_conf"])
    config["model_conf"].pop("legacy_inference", None)
    # Lang2vec conditioning is an optional part of the standard LID model.
    # Previously converted checkpoints already use the same parameter names.
    config["model"] = "espnet"
    config["loss"] = "aamsoftmax_sc_topk"
    config["loss_conf"] = dict(config["loss_conf"])
    for key in ("lang2vec_dim", "lang2vec_weight"):
        if key in config["loss_conf"]:
            config["model_conf"][key] = config["loss_conf"].pop(key)
    if source.get("model") == "lang2vec":
        # The former dedicated model enabled auxiliary training by default.
        # Keep that behavior when migrating to the disabled-by-default model.
        config["model_conf"].setdefault("lang2vec_weight", 0.2)
    feature_type = config["loss_conf"].pop("lang2vec_type", None)
    if feature_type is not None:
        if feature_type not in ("geo", "phonology_knn", "syntax_knn", "inventory_knn"):
            raise ValueError(f"Unsupported legacy lang2vec type: {feature_type}")
        config["model_conf"]["lang2vec_loss"] = (
            "mse" if feature_type == "geo" else "bce"
        )
    config["frontend"] = "s3prl"
    config["frontend_conf"] = dict(config["frontend_conf"])
    config["frontend_conf"]["frontend_conf"] = dict(
        config["frontend_conf"]["frontend_conf"]
    )
    config["frontend_conf"]["frontend_conf"]["upstream"] = "hf_wav2vec2_custom"
    config["frontend_conf"]["frontend_conf"]["path_or_url"] = "facebook/mms-1b"
    if config["frontend_conf"].get("layer") is None:
        config["frontend_conf"].pop("layer", None)
    config["lang_num"] = source["lang_num"]
    config["preprocessor"] = "lid"
    config["preprocessor_conf"] = dict(
        noise_apply_prob=0.0, rir_apply_prob=0.0, rir_scp=None
    )
    config["init"] = None
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--languages",
        type=Path,
        required=True,
        help="One ISO-639-3 code per line in classifier-index order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty")
    config = convert_config(yaml.safe_load(args.config.read_text()))
    languages = args.languages.read_text().splitlines()
    if len(languages) != config["lang_num"] or len(set(languages)) != len(languages):
        parser.error("Language count/order file must contain one unique code per class")
    source = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    converted = convert_state_dict(
        source, config["model_conf"]["lang2vec_conditioning_layers"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output_dir / "model.pth")
    (args.output_dir / "languages.txt").write_text("\n".join(languages) + "\n")
    # ModelDownloader resolves paths containing a directory separator.
    config["lang2utt"] = "./languages.txt"
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    report = dict(
        source_checkpoint=args.checkpoint.name,
        converted_not_retrained=True,
        tensor_values_changed=False,
        source_tensor_count=len(source),
        converted_tensor_count=len(converted),
        runtime=(
            "Stock S3prlFrontend and current ESPnet normalization/pooling; "
            "no legacy inference emulation"
        ),
    )
    (args.output_dir / "conversion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
