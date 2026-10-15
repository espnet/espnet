"""Inference entrypoint for ESPnet3 systems."""

import logging
import time
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.inference_runner import _load_output_fn

logger = logging.getLogger(__name__)


def infer(config: DictConfig):
    """Run configured backend inference over test datasets and write SCP files.

    input_key selects dataset columns passed as keyword arguments to the
    callable model. output_fn optionally formats native results as records
    containing the sample identifier and output fields. Writers serialize
    artifacts according to output_artifacts.
    """
    start = time.perf_counter()
    set_parallel(getattr(config, "parallel", None))

    test_sets = []
    for index, test_set in enumerate(config.dataset.test):
        name = getattr(test_set, "name", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                "inference_config.dataset.test entries must define non-empty `name` "
                f"(failed at index {index})."
            )
        test_sets.append(name)
    assert len(test_sets) > 0, "No test set found in dataset"
    assert len(test_sets) == len(set(test_sets)), "Duplicate test key found."

    logger.info(
        "Starting inference | inference_dir=%s test_sets=%s",
        getattr(config, "inference_dir", None),
        test_sets,
    )

    for test_name in test_sets:
        logger.info("===> Processing test set: %s", test_name)
        config.test_set = test_name

        output_fn_path = getattr(config, "output_fn", None)
        if output_fn_path:
            _load_output_fn(output_fn_path)

        input_key = getattr(config, "input_key", None)
        if input_key is None:
            raise RuntimeError("inference_config.input_key must be set.")
        elif isinstance(input_key, (list, tuple)) and not input_key:
            raise RuntimeError("inference_config.input_key must not be empty.")

        output_keys = getattr(config, "output_keys", None)
        if output_keys is not None:
            if isinstance(output_keys, str):
                output_keys = [output_keys]
            elif not isinstance(output_keys, (list, tuple)):
                output_keys = list(output_keys)
            if not output_keys:
                raise RuntimeError("inference_config.output_keys must not be empty.")

        idx_key = getattr(config, "idx_key", "utt_id")
        output_dir = Path(config.inference_dir) / test_name
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = getattr(config, "batch_size", None)
        provider_config = getattr(config, "provider", None)
        if provider_config is None:
            raise RuntimeError("inference_config.provider must be set.")
        raw_params = getattr(provider_config, "params", {}) or {}
        if OmegaConf.is_config(raw_params):
            provider_params = OmegaConf.to_container(raw_params, resolve=True)
        else:
            provider_params = dict(raw_params)

        provider_params["input_key"] = input_key
        provider_params["idx_key"] = idx_key
        provider_params["output_keys"] = output_keys
        if output_fn_path:
            provider_params["output_fn_path"] = output_fn_path
        artifact_configs = getattr(config, "output_artifacts", {}) or {}
        if OmegaConf.is_config(artifact_configs):
            artifact_configs = OmegaConf.to_container(artifact_configs, resolve=True)
        provider_params["output_artifacts"] = artifact_configs

        provider = instantiate(
            provider_config,
            inference_config=config,
            params=provider_params,
            _recursive_=False,
        )

        hyp_keys = output_keys if output_keys is not None else []
        runner_config = getattr(config, "runner", None)
        if runner_config is None:
            raise RuntimeError("inference_config.runner must be set.")

        runner_kwargs = {
            "provider": provider,
            "idx_key": idx_key,
            "hyp_key": hyp_keys,
            "ref_key": [],
            "batch_size": batch_size,
            "output_dir": output_dir,
        }
        runner = instantiate(runner_config, **runner_kwargs)
        if not hasattr(runner, "idx_key"):
            raise TypeError(
                f"{type(runner).__name__} must provide inference runner attributes"
            )
        dataset = provider.build_dataset(config)
        dataset_length = len(dataset)
        if dataset_length == 0:
            raise RuntimeError(
                f"Test dataset '{test_name}' is empty. "
                "Please check the dataset manifest and preprocessing outputs."
            )
        logger.info("===> Processing %d samples..", dataset_length)
        result = runner(list(range(dataset_length)))
        if isinstance(result, list):
            raise RuntimeError(
                "In-memory inference results are not supported. "
                "Use a runner that writes shard outputs via BaseRunner hooks."
            )
        if result is not True:
            raise RuntimeError(
                "Inference runner did not produce shard outputs. "
                "Use a runner that writes outputs via BaseRunner hooks."
            )
        logger.info(
            "Finished test set %s | outputs=%s",
            test_name,
            output_dir,
        )

    logger.info("Inference finished in %.2fs", time.perf_counter() - start)
