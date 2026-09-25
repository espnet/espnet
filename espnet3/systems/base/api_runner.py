"""The ``infer`` stage for a system's ``Inference``: the declaration drives the run.

With :class:`espnet3.api.inference.InferenceAPI`, a system already says what
it takes and returns. This runner reads that instead of asking the recipe:
the dataset fields to pass are the declared inputs, the model is called
through its own entry points (``model(**fields)`` for one item,
``model.batch(items)`` for a batch), every declared output is written by
what it is - text to ``<field>.scp``, audio to WAV artifacts at its own
rate, segments to JSON - and the sample id is taken from the dataset item.
So an ``inference.yaml`` needs no ``input_key``, ``output_fn`` or
``output_artifacts``::

    model:
      _target_: espnet3.systems.asr.inference.Inference
      asr_train_config: ${exp_dir}/config.yaml
      asr_model_file: ${exp_dir}/valid.acc.ave.pth
    copy:
      text: ref              # dataset columns to write beside the outputs
    runner:
      _target_: espnet3.systems.base.api_runner.APIRunner

The provider is the ordinary
:class:`espnet3.systems.base.inference_provider.InferenceProvider`: it
instantiates ``model`` on the device it picks, and that is the
``Inference``. Parallelism, resume and shard files are
:class:`espnet3.parallel.base_runner.BaseRunner`'s, as for every runner.

``output_artifacts`` still applies when set, for a file type the kind
does not imply. ``output_fn`` does not: this runner writes what the
declaration says, and refuses a configured ``output_fn`` rather than
ignore it - a recipe that needs one uses ``InferenceRunner``. ``copy``
covers the common case, the reference for scoring: ``measure`` then
reads it with ``ref_key: ref`` and the hypothesis with ``hyp_key: text``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from hydra.utils import get_class
from omegaconf import DictConfig

from espnet3.api.inference import Audio, InferenceAPI
from espnet3.systems.base.inference_runner import InferenceRunner, _iter_outputs

logger = logging.getLogger(__name__)


def declared_input_names(config: DictConfig) -> Optional[List[str]]:
    """Return the input field names the configured model class declares.

    Lets ``infer()`` default ``input_key`` from the declaration when
    ``model._target_`` names an :class:`InferenceAPI` subclass, without
    building the model. ``None`` when it names anything else or nothing.

    Args:
        config: The inference config; only ``model._target_`` is read.

    Returns:
        The declared input names, optional ones included, or ``None``.

    Examples:
        >>> cfg = OmegaConf.create(
        ...     {"model": {"_target_": "espnet3.systems.asr.inference.Inference"}}
        ... )
        >>> declared_input_names(cfg)
        ['speech']
    """
    target = getattr(getattr(config, "model", None), "_target_", None)
    if not isinstance(target, str) or not target:
        return None
    try:
        cls = get_class(target)
    except Exception:  # noqa: BLE001 - not an importable class: not ours to judge
        return None
    if isinstance(cls, type) and issubclass(cls, InferenceAPI):
        return [f.name for f in cls.inputs]
    return None


def _declared_fields(model: InferenceAPI, data: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick the declared inputs out of one dataset item."""
    fields = {}
    for f in model.inputs:
        if f.name in data:
            fields[f.name] = data[f.name]
        elif not f.optional:
            raise KeyError(
                f"dataset item has no {f.name!r}, which {type(model).__qualname__} "
                f"needs; it has {sorted(data)}"
            )
    return fields


def _record(
    output: Mapping[str, Any],
    data: Mapping[str, Any],
    idx: Any,
    idx_key: str,
    copy: Optional[Mapping[str, str]],
) -> Dict[str, Any]:
    """One result as the writers take it: id first, outputs, copied columns."""
    record: Dict[str, Any] = {idx_key: data.get(idx_key, str(idx))}
    record.update(output)
    for source, target in (copy or {}).items():
        if target in record:
            raise KeyError(
                f"copy: {target!r} is already an output (or the id); "
                "a copied column cannot replace one"
            )
        if source not in data:
            raise KeyError(
                f"copy: dataset item {record[idx_key]!r} has no {source!r} "
                f"to write as {target!r}"
            )
        record[target] = data[source]
    return record


def _writable(output: Mapping[str, Any], artifact_configs: Dict[str, dict]) -> dict:
    """Turn contract values into what the writers take; audio brings its rate."""
    out = {}
    for key, value in output.items():
        if isinstance(value, Audio):
            artifact_configs.setdefault(key, {"type": "wav", "sample_rate": value.rate})
            value = value.array
        elif isinstance(value, list):
            # the writers take no top-level list; a JSON document does
            value = {key: value}
        out[key] = value
    return out


class APIRunner(InferenceRunner):
    """Run an ``Inference`` over dataset shards and write what it declares.

    Use it as ``runner._target_`` in ``inference.yaml`` when ``model``
    names an :class:`InferenceAPI` subclass; see the module docstring for
    the config. Everything else - shards, workers, resume, the SCP files
    and their merge - is inherited from :class:`InferenceRunner`.

    Notes:
        ``forward`` stays a static method that captures nothing, as every
        runner's must, so it can run in a Dask worker.

    Examples:
        >>> runner = APIRunner(provider, output_dir="exp/decode", batch_size=8)
        >>> runner(range(len(test_set)))
        True
        >>> # exp/decode/text.scp, exp/decode/ref.scp
    """

    @staticmethod
    def forward(
        idx,
        dataset=None,
        model=None,
        idx_key: str = "utt_id",
        copy: Optional[Mapping[str, str]] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        **env,
    ):
        """Infer one item, or a batch, through the model's own entry points.

        Args:
            idx: One dataset index, or a list of them for a batch.
            dataset: Anything indexable by ``idx`` giving a mapping of
                fields.
            model: The :class:`InferenceAPI` instance the provider built.
            idx_key: The dataset field holding the sample id; the index
                itself when the item has none.
            copy: Dataset columns to write beside the outputs, source to
                target name, e.g. ``{"text": "ref"}``.
            model_kwargs: Not taken: an ``Inference`` has no call-time
                arguments; what it needs is in its constructor config.
            **env: The rest of the provider's environment, unused - except
                that a configured ``output_fn`` is refused rather than
                silently ignored.

        Returns:
            One record - ``{idx_key: ..., <outputs>..., <copied>...}`` - or a
            list of them for a batch.

        Raises:
            TypeError: If ``model`` is not an ``Inference``, ``model_kwargs``
                are given, or an ``output_fn`` is configured.
            KeyError: If an item lacks a required input or a ``copy``
                source, or a ``copy`` target is already an output.
        """
        if not isinstance(model, InferenceAPI):
            raise TypeError(
                f"APIRunner runs an Inference, not {type(model).__name__}; "
                "name an espnet3.api.inference.InferenceAPI subclass in model._target_"
            )
        if model_kwargs:
            raise TypeError(
                f"an Inference takes no call-time arguments ({sorted(model_kwargs)}); "
                "put them in its model config"
            )
        if env.get("output_fn") or env.get("output_fn_path"):
            raise TypeError(
                "APIRunner writes the declared outputs and applies no output_fn; "
                "use InferenceRunner for one, or `copy` for a dataset column"
            )
        batched = isinstance(idx, (list, tuple))
        indices = list(idx) if batched else [idx]
        items = [dataset[i] for i in indices]
        fields = [_declared_fields(model, data) for data in items]
        outputs = model.batch(fields) if batched else [model(**fields[0])]
        records = [
            _record(out, data, i, idx_key, copy)
            for out, data, i in zip(outputs, items, indices)
        ]
        return records if batched else records[0]

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Write one result, or a batch, converting contract values first."""
        writable = [
            _writable(out, writers["artifact_configs"]) for out in _iter_outputs(result)
        ]
        InferenceRunner.write_record(
            writers, writable if isinstance(result, list) else writable[0], state, **env
        )
