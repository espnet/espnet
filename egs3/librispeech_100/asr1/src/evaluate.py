"""Decoding and scoring entrypoint for the LibriSpeech 100h recipe."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.runner.base_runner import BaseRunner
from espnet3.runner.inference_provider import InferenceProvider

from .metrics import CharacterErrorRate, WordErrorRate


METRIC_REGISTRY = {
    "wer": WordErrorRate,
    "cer": CharacterErrorRate,
}


class ASRInferenceProvider(InferenceProvider):
    """Provider that builds dataset/model pairs for ASR inference."""

    @staticmethod
    def build_dataset(cfg: DictConfig):
        organizer = instantiate(cfg.dataset)
        test_name = cfg.runtime.test_name
        if not hasattr(organizer, "test_sets"):
            raise RuntimeError("DataOrganizer is expected to expose test_sets for decoding")
        return organizer.test_sets[test_name]

    @staticmethod
    def build_model(cfg: DictConfig):
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        return instantiate(model_cfg)


class ASRInferenceRunner(BaseRunner):
    """Run inference over LibriSpeech splits."""

    @staticmethod
    def forward(idx: int, *, dataset, model, **env) -> Dict:
        sample = dataset[idx]
        speech = sample["speech"]
        sample_rate = sample.get("sample_rate", 16_000)
        utt_id = sample.get("utt_id", f"utt{idx}")
        ref = sample.get("text", "")

        start = time.time()
        results = model(speech)
        end = time.time()

        if isinstance(results, tuple):
            nbest = results[0]
        else:
            nbest = results
        hypothesis = nbest[0][0] if nbest else ""

        duration = len(speech) / float(sample_rate)
        elapsed = end - start
        rtf = elapsed / duration if duration > 0 else 0.0

        return {
            "utt_id": utt_id,
            "hypothesis": hypothesis,
            "ref": ref,
            "rtf": rtf,
        }


def _make_runner_config(cfg: DictConfig, test_name: str) -> DictConfig:
    runtime = OmegaConf.create({"test_name": test_name, "device": cfg.runtime.device})
    composed = OmegaConf.create({"dataset": cfg.dataset, "model": cfg.model, "runtime": runtime})
    return composed


def _write_transcripts(test_name: str, outputs: List[Dict], decode_dir: Path) -> Dict[str, List[str]]:
    test_dir = decode_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    hyp_path = test_dir / "hyp.trn"
    ref_path = test_dir / "ref.trn"
    meta_path = test_dir / "meta.jsonl"

    lines = {"hypothesis": [], "ref": [], "utt_id": []}

    with hyp_path.open("w", encoding="utf-8") as hyp_f, ref_path.open(
        "w", encoding="utf-8"
    ) as ref_f, meta_path.open("w", encoding="utf-8") as meta_f:
        for entry in outputs:
            utt_id = entry["utt_id"]
            hyp = entry["hypothesis"].strip()
            ref = entry["ref"].strip()
            lines["utt_id"].append(utt_id)
            lines["hypothesis"].append(hyp)
            lines["ref"].append(ref)
            hyp_f.write(f"{utt_id} {hyp}\n")
            ref_f.write(f"{utt_id} {ref}\n")
            meta_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return lines


def _resolve_metric(metric_cfg: DictConfig):
    if "metric" in metric_cfg:
        return instantiate(metric_cfg.metric), metric_cfg.get("apply_to")

    name = metric_cfg.get("name")
    if name is None:
        raise ValueError("Metric entry must define 'name' or 'metric' with _target_.")
    params = metric_cfg.get("params", {})
    metric_cls = METRIC_REGISTRY.get(name.lower())
    if metric_cls is None:
        raise KeyError(f"Unknown metric: {name}")
    metric = metric_cls(**OmegaConf.to_container(params, resolve=True))
    return metric, metric_cfg.get("apply_to")


def decode(cfg: DictConfig) -> Dict:
    if cfg.get("parallel"):
        set_parallel(cfg.parallel)

    decode_dir = Path(cfg.decode_dir)
    decode_dir.mkdir(parents=True, exist_ok=True)

    organizer = instantiate(cfg.dataset)
    results_by_metric: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    transcripts: Dict[str, Dict[str, List[str]]] = {}

    for test_name, dataset in organizer.test_sets.items():
        runner_cfg = _make_runner_config(cfg, test_name)
        provider = ASRInferenceProvider(runner_cfg)
        runner = ASRInferenceRunner(provider)
        indices = list(range(len(dataset)))
        outputs = runner(indices)
        transcripts[test_name] = _write_transcripts(test_name, outputs, decode_dir)
        print(f"Decoded {test_name}: {len(outputs)} utterances")

    metrics = cfg.get("metrics", [])
    for metric_cfg in metrics:
        metric, apply_to = _resolve_metric(metric_cfg)
        target_sets = apply_to or list(transcripts.keys())
        metric_name = metric_cfg.get("name") or metric.__class__.__name__
        for test_name in target_sets:
            scores = metric(transcripts[test_name], test_name, decode_dir)
            results_by_metric[metric_name][test_name] = scores

    result_path = decode_dir / "metrics_summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results_by_metric, f, indent=2, ensure_ascii=False)

    print("\n===== Score Summary =====")
    for metric_name, test_scores in results_by_metric.items():
        print(f"Metric: {metric_name}")
        for test_name, scores in test_scores.items():
            print(f"  [{test_name}]")
            for key, value in scores.items():
                print(f"    {key}: {value}")
    print("=========================")

    return results_by_metric


def debug_sample(cfg: DictConfig) -> None:
    organizer = instantiate(cfg.dataset)
    test_name = cfg.runtime.get("debug_test") or next(iter(organizer.test_sets))
    dataset = organizer.test_sets[test_name]
    runner_cfg = _make_runner_config(cfg, test_name)
    provider = ASRInferenceProvider(runner_cfg)
    env = provider.build_env_local()
    sample = ASRInferenceRunner.forward(0, **env)
    print(f"Debug sample from {test_name}:")
    print(json.dumps(sample, ensure_ascii=False, indent=2))


def main(config_name: str = "evaluate", overrides: Iterable[str] | None = None) -> None:
    overrides = list(overrides or [])
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)

    if cfg.runtime.get("debug_sample"):
        debug_sample(cfg)
    else:
        decode(cfg)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="evaluate", help="Name of the Hydra config to load")
    parser.add_argument("overrides", nargs="*", default=[], help="Hydra-style overrides")
    args = parser.parse_args()
    main(config_name=args.config, overrides=args.overrides)