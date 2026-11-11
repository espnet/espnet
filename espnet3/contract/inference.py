"""Config-declared input and output contracts for inference."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from importlib import import_module
from typing import Any

from omegaconf import DictConfig, OmegaConf

from espnet3.contract.field import Field
from espnet3.contract.kinds import KINDS


@dataclass(frozen=True)
class InferenceContract:
    """Input and output declarations shared by all inference entry points."""

    inputs: tuple[Field, ...]
    outputs: tuple[Field, ...]
    imports: tuple[str, ...] = ()

    def prepare_input(self, sample: Mapping[str, Any], model: Any) -> dict[str, Any]:
        """Validate and normalize one copy of a sample before batching.

        Args:
            sample: One dataset item holding the declared input fields.
            model: The backend, for what it knows (its ``sample_rate``).

        Returns:
            A copy of ``sample`` whose declared inputs are checked and
            converted to what the backend receives.

        Raises:
            TypeError: If ``sample`` is not a mapping.
            KeyError: If a required input is missing.

        Examples:
            >>> contract = load_inference_contract(
            ...     {
            ...         "inputs": {"text": {"kind": "text"}},
            ...         "outputs": {"text": {"kind": "text"}},
            ...     }
            ... )
            >>> contract.prepare_input({"text": "hi"}, model=None)
            {'text': 'hi'}
        """
        if not isinstance(sample, Mapping):
            raise TypeError("inference input must be a mapping")
        prepared = dict(sample)
        for spec in self.inputs:
            if spec.name not in sample or (spec.optional and sample[spec.name] is None):
                if spec.optional:
                    prepared.pop(spec.name, None)
                    continue
                raise KeyError(f"Required inference input {spec.name!r} is missing")
            value = KINDS[spec.kind].apply(
                sample[spec.name], spec, model, spec.options, output=False
            )
            prepared[spec.name] = KINDS[spec.kind].to_model_input(value, spec)
        return prepared

    def validate_input_keys(self, input_key: Any) -> None:
        """Ensure every backend argument has one declared input field.

        Args:
            input_key: The configured ``input_key``: one name or a list.

        Returns:
            None.

        Raises:
            ValueError: If ``input_key`` is missing or differs from the declared
                inputs.

        Examples:
            >>> contract = load_inference_contract(
            ...     {
            ...         "inputs": {"text": {"kind": "text"}},
            ...         "outputs": {"text": {"kind": "text"}},
            ...     }
            ... )
            >>> contract.validate_input_keys("text")
            >>> contract.validate_input_keys("speech")
            Traceback (most recent call last):
            ValueError: inference input_key must match contract.inputs; ...
        """
        if input_key is None:
            raise ValueError("inference input_key is required by the contract")
        keys = [input_key] if isinstance(input_key, str) else list(input_key)
        declared = [spec.name for spec in self.inputs]
        if len(keys) != len(set(keys)) or set(keys) != set(declared):
            raise ValueError(
                "inference input_key must match contract.inputs; "
                f"input_key={keys}, contract.inputs={declared}"
            )

    def prepare_output(
        self, result: Any, model: Any, *, idx: Any, idx_key: str, sample: Mapping
    ) -> dict[str, Any]:
        """Validate named backend outputs and attach the sample identifier.

        Args:
            result: What the backend returned: a mapping of output names.
            model: The backend, for what it knows (its ``sample_rate``).
            idx: The dataset index of the sample.
            idx_key: The name under which the sample identifier is attached.
            sample: The original dataset item, a source for ``idx_key``.

        Returns:
            The checked outputs plus the sample identifier under ``idx_key``.

        Raises:
            TypeError: If ``result`` is not a mapping.
            ValueError: If ``result`` has an undeclared output.
            KeyError: If a declared output is missing.

        Examples:
            >>> contract = load_inference_contract(
            ...     {
            ...         "inputs": {"text": {"kind": "text"}},
            ...         "outputs": {"text": {"kind": "text"}},
            ...     }
            ... )
            >>> contract.prepare_output(
            ...     {"text": "hi"}, None, idx=0, idx_key="utt_id", sample={}
            ... )
            {'text': 'hi', 'utt_id': '0'}
        """
        if not isinstance(result, Mapping):
            raise TypeError(
                "A backend with an inference contract must return a mapping"
            )
        expected = {spec.name for spec in self.outputs}
        if idx_key in expected:
            raise ValueError(f"{idx_key!r} is reserved for the sample identifier")
        extra = set(result) - expected - {idx_key}
        if extra:
            raise ValueError(f"Undeclared inference output(s): {sorted(extra)}")
        prepared = {}
        for spec in self.outputs:
            if spec.name not in result:
                raise KeyError(f"Required inference output {spec.name!r} is missing")
            prepared[spec.name] = KINDS[spec.kind].apply(
                result[spec.name], spec, model, spec.options, output=True
            )
        prepared[idx_key] = result.get(idx_key, sample.get(idx_key, str(idx)))
        return prepared


def _fields(raw: Mapping[str, Any], section: str) -> tuple[Field, ...]:
    """Build the ``Field`` declarations of one contract section."""
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"inference contract.{section} must be a nonempty mapping")
    fields = []
    for name, options in raw.items():
        if not isinstance(options, Mapping):
            raise TypeError(f"contract.{section}.{name} must be a mapping")
        field_keys = {item.name for item in dataclass_fields(Field)} - {
            "name",
            "options",
        }
        declaration = {
            key: value for key, value in options.items() if key in field_keys
        }
        declaration.setdefault("kind", None)
        kind_options = {
            key: value for key, value in options.items() if key not in field_keys
        }
        fields.append(Field(str(name), **declaration, options=kind_options))
    if section == "outputs" and any(spec.optional for spec in fields):
        raise ValueError("inference contract outputs cannot be optional")
    return tuple(fields)


def load_inference_contract(
    config: DictConfig | Mapping[str, Any] | None, *, allow_imports: bool = True
) -> InferenceContract | None:
    """Build a contract from the inference config after loading custom kinds.

    Args:
        config: The ``contract`` section: a mapping (or ``DictConfig``) with
            ``inputs``, ``outputs`` and optional ``imports``; ``None`` for none.
        allow_imports: Whether ``imports`` may be loaded; ``False`` for an
            untrusted bundle.

    Returns:
        The contract, or ``None`` when ``config`` is ``None``.

    Raises:
        TypeError: If ``config`` or ``imports`` has the wrong type.
        ValueError: If the contract has an unknown option, or ``imports`` is set
            while ``allow_imports`` is false.

    Examples:
        >>> contract = load_inference_contract(
        ...     {
        ...         "inputs": {"text": {"kind": "text"}},
        ...         "outputs": {"text": {"kind": "text"}},
        ...     }
        ... )
        >>> [field.name for field in contract.inputs]
        ['text']
    """
    if config is None:
        return None
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError("inference contract must be a mapping")
    unknown = set(config) - {"imports", "inputs", "outputs"}
    if unknown:
        raise ValueError(f"Unknown inference contract option(s): {sorted(unknown)}")
    raw_imports = config.get("imports", ()) or ()
    if isinstance(raw_imports, str):
        raise TypeError("contract.imports must be a list of module names")
    imports = tuple(raw_imports)
    if any(not isinstance(module, str) or not module for module in imports):
        raise TypeError("contract.imports must contain module names")
    if imports and not allow_imports:
        raise ValueError("contract.imports requires trust_user_code=True")
    for module in imports:
        import_module(module)
    return InferenceContract(
        inputs=_fields(config.get("inputs"), "inputs"),
        outputs=_fields(config.get("outputs"), "outputs"),
        imports=imports,
    )


def resolve_inference_contract(contract, root=None):
    """Load a serialized contract and its recipe registrations.

    Args:
        contract: An ``InferenceContract``, its mapping form, or ``None``.
        root: A directory added to ``sys.path`` before loading a mapping, so its
            ``imports`` resolve.

    Returns:
        The contract, or ``None`` when ``contract`` is ``None``.

    Raises:
        TypeError: If ``contract`` is none of the accepted forms.

    Examples:
        >>> resolve_inference_contract(None) is None
        True
        >>> contract = resolve_inference_contract(
        ...     {
        ...         "inputs": {"text": {"kind": "text"}},
        ...         "outputs": {"text": {"kind": "text"}},
        ...     }
        ... )
        >>> [field.name for field in contract.outputs]
        ['text']
    """
    if contract is None:
        return None
    if isinstance(contract, Mapping):
        if root and root not in sys.path:
            sys.path.insert(0, root)
        contract = load_inference_contract(contract)
    if not isinstance(contract, InferenceContract):
        raise TypeError("inference_contract must be an InferenceContract")
    return contract


def prepare_inference_inputs(dataset, idx, contract, model):
    """Return prepared and original sample mappings for one runner call.

    Args:
        dataset: The dataset the runner reads from.
        idx: One index or a list of indices.
        contract: The inference contract.
        model: The backend, for what it knows (its ``sample_rate``).

    Returns:
        Two dicts keyed by dataset index: the prepared samples and the originals.

    Examples:
        >>> contract = load_inference_contract(
        ...     {
        ...         "inputs": {"text": {"kind": "text"}},
        ...         "outputs": {"text": {"kind": "text"}},
        ...     }
        ... )
        >>> prepare_inference_inputs([{"text": "hi"}], 0, contract, None)
        ({0: {'text': 'hi'}}, {0: {'text': 'hi'}})
    """
    indices = list(idx) if isinstance(idx, (list, tuple)) else [idx]
    original = {index: dataset[index] for index in indices}
    prepared = {
        index: contract.prepare_input(sample, model)
        for index, sample in original.items()
    }
    return prepared, original


def validate_inference_outputs(result, idx, contract, model, original, idx_key):
    """Validate one named output mapping for each input sample.

    Args:
        result: What the backend returned for ``idx``.
        idx: One index or a list of indices.
        contract: The inference contract.
        model: The backend, for what it knows (its ``sample_rate``).
        original: The original samples keyed by dataset index.
        idx_key: The name under which the sample identifier is attached.

    Returns:
        One checked output mapping, or a list of them for a batch.

    Raises:
        ValueError: If the backend did not return one mapping per sample.

    Examples:
        >>> contract = load_inference_contract(
        ...     {
        ...         "inputs": {"text": {"kind": "text"}},
        ...         "outputs": {"text": {"kind": "text"}},
        ...     }
        ... )
        >>> validate_inference_outputs(
        ...     {"text": "hi"}, 0, contract, None, {0: {}}, "utt_id"
        ... )
        {'text': 'hi', 'utt_id': '0'}
    """
    indices = list(idx) if isinstance(idx, (list, tuple)) else [idx]
    if isinstance(idx, (list, tuple)) and isinstance(result, Mapping):
        if all(
            isinstance(value, (list, tuple)) and len(value) == len(indices)
            for value in result.values()
        ):
            result = [
                {key: values[position] for key, values in result.items()}
                for position in range(len(indices))
            ]
    results = result if isinstance(result, list) else [result]
    if len(results) != len(indices):
        raise ValueError(
            "Contracted inference must return one mapping per input sample"
        )
    outputs = [
        contract.prepare_output(
            value,
            model,
            idx=index,
            idx_key=idx_key,
            sample=original[index],
        )
        for index, value in zip(indices, results)
    ]
    return outputs if isinstance(idx, (list, tuple)) else outputs[0]
