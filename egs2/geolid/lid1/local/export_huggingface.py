#!/usr/bin/env python3
# Copyright 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Convert a released GeoLID checkpoint to a self-contained Transformers model."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

from espnet2.lid.huggingface.configuration_geolid import GeoLIDConfig
from espnet2.lid.huggingface.modeling_geolid import GeoLIDForAudioClassification


def convert_state_dict(state):
    """Rename the upstream and remove only verified duplicate parameter aliases."""
    result = {}
    for key, value in state.items():
        if key.startswith("upstream_encoder."):
            canonical = (
                "frontend.upstream.upstream.model.encoder."
                + key[len("upstream_encoder.") :]
            )
            if canonical not in state or not torch.equal(value, state[canonical]):
                raise ValueError(f"Conflicting checkpoint alias: {key}")
            continue
        if ".aamsoftmax_loss." in key:
            canonical = "loss." + key.split(".aamsoftmax_loss.", 1)[1]
            if canonical not in state or not torch.equal(value, state[canonical]):
                raise ValueError(f"Conflicting loss alias: {key}")
            continue
        if ".encoder.conditioning_projs." in key:
            prefix, suffix = key.split(".encoder.conditioning_projs.", 1)
            layer, parameter = suffix.split(".", 1)
            key = f"{prefix}.encoder.lang2vec_conditioning_projs.{layer}.0.{parameter}"
        if key.startswith("frontend.upstream.upstream.model."):
            key = "wav2vec2." + key[len("frontend.upstream.upstream.model.") :]
        elif key == "frontend.featurizer.weights":
            key = "layer_weights"
        result[key] = value
    return result


def export(checkpoint, training_config, output_dir, wav2vec2_config, lang2utt=None):
    """Export without retraining, quantization or changing tensor values."""
    with open(training_config) as stream:
        args = yaml.safe_load(stream)
    supported = {
        "model": "upstream_condition",
        "frontend": "s3prl_condition",
        "encoder": "ecapa_tdnn",
        "pooling": "chn_attn_stat",
        "projector": "rawnet3",
        "encoder_condition": "identity",
        "pooling_condition": "chn_attn_stat",
        "projector_condition": "rawnet3",
        "normalize": "utterance_mvn",
        "loss": "aamsoftmax_sc_topk_lang2vec",
    }
    for key, expected in supported.items():
        if args.get(key) != expected:
            raise ValueError(f"Only {key}={expected} is supported by this exporter.")
    front = args["frontend_conf"]
    if not front.get("multilayer_feature") or front.get("layer") is not None:
        raise ValueError("The exporter requires all-layer weighted features.")
    if (
        front.get("frontend_conf", {}).get("normalize", False)
        or front.get("frontend_conf", {}).get("tile_factor", 1) != 1
    ):
        raise ValueError("Normalized or tiled upstream features are not supported.")
    if args.get("normalize_conf", {}).get("norm_vars", False) or not args.get(
        "normalize_conf", {}
    ).get("norm_means", True):
        raise ValueError("The exporter requires mean-only utterance normalization.")
    if not args["model_conf"].get("apply_intermediate_lang2vec_condition", True):
        raise ValueError("The exporter requires intermediate conditioning.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    labels = None
    if lang2utt is not None:
        labels = [
            line.split()[0]
            for line in Path(lang2utt).read_text().splitlines()
            if line.strip()
        ]
        if len(labels) != args["lang_num"] or len(set(labels)) != len(labels):
            raise ValueError("lang2utt must contain one unique label per output class.")
    base = Wav2Vec2Config.from_pretrained(wav2vec2_config).to_dict()
    config = GeoLIDConfig(
        wav2vec2_config=base,
        conditioning_layers=args["model_conf"]["lang2vec_conditioning_layers"],
        shared_conditioning_proj=args["model_conf"].get(
            "shared_conditioning_proj", False
        ),
        lang2vec_dim=args["loss_conf"]["lang2vec_dim"],
        conditioning_embedding_dim=args.get("projector_condition_conf", {}).get(
            "output_size", 192
        ),
        encoder_conf=args["encoder_conf"],
        embedding_dim=args["projector_conf"]["output_size"],
        subcenters=args["loss_conf"].get("K", 3),
        scale=args["loss_conf"].get("scale", 32.0),
        num_labels=args["lang_num"],
        id2label=None if labels is None else dict(enumerate(labels)),
        label2id=(
            None if labels is None else {label: i for i, label in enumerate(labels)}
        ),
        label_mapping_verified=labels is not None,
        legacy_inference=args["model_conf"].get("legacy_inference", True),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    converted = convert_state_dict(state)
    with torch.device("meta"):
        model = GeoLIDForAudioClassification(config)
    # Strict loading rejects missing, extra, and incorrectly shaped tensors.
    model.load_state_dict(converted, strict=True, assign=True)
    model.eval()
    model.save_pretrained(output_dir, safe_serialization=True)
    Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    ).save_pretrained(output_dir)
    report = {
        "source_checkpoint": Path(checkpoint).name,
        "source_tensor_count": len(state),
        "exported_tensor_count": len(converted),
        "label_mapping_verified": labels is not None,
        "weights_changed": False,
    }
    (output_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return model


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wav2vec2-config", default="facebook/mms-1b")
    parser.add_argument(
        "--lang2utt",
        help="Original training label order; otherwise retain numeric LABEL_i names.",
    )
    return parser


if __name__ == "__main__":
    export(**vars(get_parser().parse_args()))
