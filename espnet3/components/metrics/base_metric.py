"""Base metric interfaces for ESPnet3."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, Iterator, Tuple

from espnet3.api.inference import Field
from espnet3.components.contract.metrics import check_metric_contract


class BaseMetric(ABC):
    """Base class for metrics that consume inference output paths.

    A subclass declares ``inputs``/``outputs`` (the same
    :class:`~espnet3.api.inference.Field` declaration
    :class:`~espnet3.api.inference.InferenceAPI` uses): ``inputs`` names
    what the metric reads (checked against the configured model's own
    declared outputs, via
    ``espnet3.components.contract.metrics.check_metric_inputs``), and
    ``outputs`` names the keys of the result dict, each of kind
    ``"number"``. There is no undeclared fallback: a metric that declares
    neither raises ``TypeError`` as soon as it is instantiated.

    A metric whose inputs/outputs never change (most of them - ``WER``,
    ``CER``, ``TER``) declares them as class attributes. A metric whose
    contract instead depends on its own configuration (a VERSA-style
    metric, whose ``score_config`` names which measures to compute) sets
    ``self.inputs``/``self.outputs`` in its own ``__init__`` instead; the
    instance's own attributes take priority over the class's, and the
    contract is checked once construction finishes (not at class
    definition, which a configuration-dependent contract cannot satisfy)
    - a subclass that does this must call ``super().__init__()`` after
    setting them, so the check sees the instance's own declaration.

    Examples:
        A fixed contract (``WER``-style):

        >>> class ExampleMetric(BaseMetric):
        ...     inputs = (Field("ref", "text"), Field("hyp", "text"))
        ...     outputs = (Field("score", "number"),)
        ...     def __call__(self, data, test_name, output_dir):
        ...         return {"score": 0.0}
        >>> ExampleMetric().inputs[0]
        Field(name='ref', kind='text', label='Ref', optional=False, channels=1)

        A contract built from the instance's own configuration
        (VERSA-style): ``ref`` is optional (not every VERSA measure needs
        a reference), and the output keys follow ``score_config``:

        >>> class ExampleVersaMetric(BaseMetric):
        ...     def __init__(self, score_config):
        ...         self.score_config = score_config
        ...         self.inputs = (
        ...             Field("hyp", "text"),
        ...             Field("ref", "text", optional=True),
        ...         )
        ...         self.outputs = tuple(Field(name, "number") for name in score_config)
        ...         super().__init__()
        ...     def __call__(self, data, test_name, output_dir):
        ...         return {name: 0.0 for name in self.score_config}
        >>> [f.name for f in ExampleVersaMetric(["mcd", "f0"]).outputs]
        ['mcd', 'f0']
    """

    #: What this metric reads: one Field per SCP input. The name is the
    #: key of ``data`` (and the default SCP file name); the kind says what
    #: the SCP values hold (``text``: the value itself, ``audio``: a path).
    #: A class attribute for a fixed contract; set on ``self`` instead,
    #: before calling ``super().__init__()``, for one built from the
    #: instance's own configuration.
    inputs: ClassVar[Tuple[Field, ...]]
    #: What this metric returns: one Field per key of the result dict, all
    #: of kind ``number``. Same class-attribute-or-``self`` choice as
    #: :attr:`inputs`.
    outputs: ClassVar[Tuple[Field, ...]]

    def __init__(self) -> None:
        """Check this instance's declared contract.

        Reads ``self.inputs``/``self.outputs``, so it sees a fixed class
        attribute or an instance attribute a subclass's own ``__init__``
        set before calling this (``super().__init__()``), whichever the
        subclass uses.

        Note:
            This is a convenience, not the only place the contract is
            checked: a subclass that overrides ``__init__`` without
            calling ``super().__init__()`` skips it, so ``measure()``
            calls ``check_metric_contract`` again on every metric it
            instantiates, trusting no declaration this never ran on.

        Raises:
            TypeError: Neither ``inputs`` nor ``outputs`` is declared, or
                the declaration is malformed.
        """
        check_metric_contract(self)

    @abstractmethod
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        """Compute metrics for an inference test set.

        Args:
            data (Dict[str, Path]): Mapping of input to metric input
                files. The most common case is SCP inputs such as ``ref.scp``
                and ``hyp.scp``, but callers may also provide other file or
                directory paths for metrics that invoke external tools.
                Concrete metrics may stream SCP contents, materialize lists, or
                pass paths directly to subprocesses. For example:

                .. code-block:: python

                    for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
                        ref = row["ref"]
                        hyp = row["hyp"]

                The keys are taken from inference-time SCP files and should match
                what the concrete metric class expects (e.g., ``ref``/``hyp``).
                To add extra inputs (e.g., a ``prompt`` field), define them in
                the metrics config ``inputs`` and provide a matching SCP file:

                .. code-block:: yaml

                    metrics:
                      - metric:
                          _target_: espnet3.systems.esp2_asr.metrics.wer.WER
                          clean_types:
                        inputs:
                          ref: ref
                          hyp: hyp
                          prompt: prompt

                This exposes ``inference_dir/<test_name>/prompt.scp`` as
                ``data["prompt"]`` for direct use.
            test_name (str): Name of the test dataset (e.g., "test-other"). This
                corresponds to the test set name defined by the data organizer.
            output_dir (Path): Root path where hypothesis/reference files are stored.

        Returns:
            Dict[str, float]: Computed metric result(s).

        Example:
            A metric that consumes aligned reference/hypothesis text can use:

            >>> for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
            ...     ref = row["ref"]
            ...     hyp = row["hyp"]

            A metric backed by an external CLI can instead use:

            >>> ref_path = data["ref"]
            >>> hyp_dir = data["hyp_dir"]
        """
        raise NotImplementedError

    def iter_inputs(
        self, data: Dict[str, Path], *keys: str
    ) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Yield rows from one or more SCP inputs with shared utterance IDs.

        This helper reads one or more SCP files in lockstep. For a single key,
        it behaves as a streaming SCP iterator that returns ``(utt_id, row)``
        pairs. For multiple keys, it additionally validates that all files
        contain the same utterance IDs in the same order. It is intended for
        metrics that want a single streaming API regardless of how many input
        files they consume.

        Args:
            data (Dict[str, Path]): Mapping from input aliases to SCP paths.
            *keys (str): Input aliases to read together, e.g.
                ``("ref",)`` or ``("ref", "hyp")``.

        Yields:
            Iterator[Tuple[str, Dict[str, str]]]:
                Tuples of ``(utt_id, row)``, where ``row`` is an alias -> value
                mapping for that utterance.

                .. code-block:: python

                    (
                        "utt1",
                        {"ref": "the cat", "hyp": "the bat"},
                    )

        Raises:
            AssertionError: If no keys are provided, if one file has more
                entries than another, or if utterance IDs do not match across
                files.

        Example:
            >>> for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
            ...     print(utt_id, row["ref"], row["hyp"])
            utt1 the cat the bat
            utt2 a dog a dog

            >>> for utt_id, row in self.iter_inputs(data, "ref"):
            ...     print(utt_id, row["ref"])
            utt1 the cat
            utt2 a dog

        Notes:
            - Alignment is checked in file order.
            - This helper does not sort utterance IDs.
            - Each input is expected to be in SCP format:

              .. code-block:: text

                  utt1 value
                  utt2 another value
        """
        assert keys, "At least one SCP key is required"

        files = {}
        try:
            for key in keys:
                files[key] = open(data[key], "r", encoding="utf-8")
            iterators = {
                key: self._iter_scp_file(file_obj) for key, file_obj in files.items()
            }

            while True:
                rows = {}
                finished_keys = []

                # Pull one row from each input to keep all SCP files in lockstep.
                for key in keys:
                    try:
                        rows[key] = next(iterators[key])
                    except StopIteration:
                        finished_keys.append(key)

                # Stop only when every input is exhausted at the same time.
                if len(finished_keys) == len(keys):
                    break

                # If only some inputs ended, the SCP files have different lengths.
                assert not finished_keys, f"SCP length mismatch across keys: {keys}"

                # Use the first key as the reference utt_id for this aligned row.
                utt_id = rows[keys[0]][0]
                # Compare the remaining keys against the reference key above.
                for key in keys[1:]:
                    assert rows[key][0] == utt_id, (
                        f"UID mismatch between {keys[0]} and {key}: "
                        f"{utt_id} != {rows[key][0]}"
                    )
                yield utt_id, {key: rows[key][1] for key in keys}
        finally:
            for file_obj in files.values():
                file_obj.close()

    def _iter_scp_file(self, file_obj) -> Iterator[Tuple[str, str]]:
        """Yield ``(utt_id, value)`` pairs from an opened SCP file."""
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            yield utt_id, value
