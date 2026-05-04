"""Inference entrypoint for ESPnet3 systems."""

import logging
import time
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.inference_runner import _load_output_fn
from espnet3.utils.writer_utils import write_artifact

logger = logging.getLogger(__name__)


def _flatten_results(results):
    flat = []
    for item in results:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _collect_scp_lines(results, idx_key: str, hyp_keys, ref_keys):
    scp_lines = {}
    hyp_keys = list(hyp_keys) if isinstance(hyp_keys, (list, tuple)) else [hyp_keys]
    ref_keys = list(ref_keys) if isinstance(ref_keys, (list, tuple)) else [ref_keys]

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                f"Expected dict output, got {type(result).__name__}: {result}"
            )

        idx_value = result[idx_key]
        if isinstance(idx_value, (list, tuple)):
            raise TypeError(
                f"'{idx_key}' must be a scalar, got {type(idx_value).__name__}"
            )

        for field_key in (*ref_keys, *hyp_keys):
            value = result[field_key]
            if isinstance(value, (list, tuple)):
                raise TypeError(
                    f"Top-level list outputs are not supported for '{field_key}'. "
                    "Return a single value per field, or wrap structured content in a "
                    "dict so it can be saved as JSON."
                )
            scp_lines.setdefault(field_key, []).append(f"{idx_value} {value}")

    return scp_lines


def _is_scp_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def _materialize_output_value(
    *,
    idx_value,
    field_key: str,
    value,
    output_dir: Path,
    artifact_config: dict | None,
):
    if _is_scp_scalar(value):
        return value

    if isinstance(value, (list, tuple)):
        raise TypeError(
            f"Top-level list outputs are not supported for '{field_key}'. "
            "Return a single value per field, or wrap structured content in a "
            "dict so it can be saved as JSON."
        )

    artifact_dir = output_dir / field_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / str(idx_value)
    artifact_path = write_artifact(value, target, field_config=artifact_config)
    return artifact_path.as_posix()


def infer(config: DictConfig):
    """Run inference over all configured test sets and write SCP files.

    This entrypoint expects each inference result to be a dict containing a
    sample identifier under ``utt_id`` (or the configured ``idx_key``), plus
    one value per output field. The final SCP files are always written under:

    .. code-block:: text

        ${inference_dir}/<test_name>/<field>.scp

    where each line is:

    .. code-block:: text

        <utt_id> <value_or_path>

    Primitive values are written directly into SCP files.

    Example return value from ``output_fn`` or directly from the inference
    model:

    .. code-block:: python

        {"utt_id": "utt1", "hyp": "hello world"}

    Generated SCP:

    .. code-block:: text

        inference_dir/test-clean/hyp.scp
          utt1 hello world

    Non-scalar values are serialized through
    :func:`espnet3.utils.writer_utils.write_artifact`. That function documents
    the detailed rules for JSON, NPY, pickle, WAV, and custom writer cases.

    Example WAV configuration:

    .. code-block:: yaml

        output_artifacts:
          audio:
            type: wav
            sample_rate: 16000

    Example return value from ``output_fn``:

    .. code-block:: python

        {"utt_id": "utt1", "audio": waveform_numpy}

    Top-level ``list`` / ``tuple`` outputs are not supported. Each output
    field must be a single value. If you need structured content, return a
    ``dict`` and let it be serialized as JSON.

    Args:
        config: Hydra/OmegaConf configuration containing the dataset,
            inference directory, provider/runner definitions, and optional
            ``output_artifacts`` writer settings.
    """
    start = time.perf_counter()
    set_parallel(config.parallel)

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

        if isinstance(input_key, (list, tuple)) and not input_key:
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
        if output_fn_path:
            provider_params["output_fn_path"] = output_fn_path

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
            "async_mode": False,
            "idx_key": idx_key,
            "hyp_key": hyp_keys,
            "ref_key": [],
            "batch_size": batch_size,
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
        out = runner(list(range(dataset_length)))
        if out is None:
            raise RuntimeError("Async inference is not supported in this entrypoint.")
        # Runner can return nested lists. normalize to flat list.
        results = _flatten_results(out)
        if not results:
            raise RuntimeError("No inference results available.")
        first = results[0]
        resolved_idx_key = runner.resolve_idx_key(first)
        if output_keys is None:
            output_keys = [key for key in first.keys() if key != resolved_idx_key]
            if not output_keys:
                raise RuntimeError("No output keys found in inference results.")

        output_dir = Path(config.inference_dir) / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_configs = getattr(config, "output_artifacts", {}) or {}
        if OmegaConf.is_config(artifact_configs):
            artifact_configs = OmegaConf.to_container(artifact_configs, resolve=True)

        for result in results:
            idx_value = result[resolved_idx_key]
            for key, value in list(result.items()):
                if key == resolved_idx_key:
                    continue
                result[key] = _materialize_output_value(
                    idx_value=idx_value,
                    field_key=key,
                    value=value,
                    output_dir=output_dir,
                    artifact_config=artifact_configs.get(key),
                )

        # Convert output dicts into per-key SCP lines (utt_id + value).
        scp_lines = _collect_scp_lines(
            results,
            idx_key=resolved_idx_key,
            hyp_keys=output_keys,
            ref_keys=[],
        )

        # create scp files
        for key, lines in scp_lines.items():
            with open(output_dir / f"{key}.scp", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        logger.info(
            "Finished test set %s | outputs=%s",
            test_name,
            output_dir,
        )

    logger.info("Inference finished in %.2fs", time.perf_counter() - start)
