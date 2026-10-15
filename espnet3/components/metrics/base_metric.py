"""Base metric interfaces for ESPnet3."""

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

#: How a state combines across chunks and ranks: ``"sum"`` adds numbers,
#: ``"cat"`` concatenates lists.
Reduce = Literal["sum", "cat"]
#: A state's value: plain Python only, never a tensor.
StateValue = Union[int, float, List[Any]]


class BaseMetric(ABC):
    """Accumulate metric state and score aligned SCP inputs.

    The metrics config maps keyword arguments of update() to SCP files or
    dataset columns. Subclasses implement update() and compute(); no schema
    declaration or inference-model introspection is required.
    """

    #: Rows per :meth:`update` call inside :meth:`__call__`.
    chunk_size: ClassVar[int] = 1000
    #: True only while :meth:`__call__` runs with an ``output_dir``. ``update``
    #: may keep per-utterance text for :meth:`write_artifacts` only when it is
    #: True; a plain ``update``/``compute`` (a training callback) leaves it False
    #: and keeps no text. Not a state: it never appears in :meth:`state`.
    collect_artifacts: bool = False

    def add_state(self, name: str, default: StateValue, reduce: Reduce) -> None:
        """Register a state and set ``self.<name>`` to a copy of ``default``.

        May be called before or after ``super().__init__()``; the registry
        is created by the first call.

        Args:
            name: Attribute name of the state.
            default: The fresh value: an ``int`` or ``float`` for ``"sum"``,
                a ``list`` for ``"cat"``. :meth:`reset` restores a copy.
            reduce: ``"sum"`` or ``"cat"``, used by :meth:`merge_states`.

        Raises:
            ValueError: ``name`` is registered or already an attribute,
                ``reduce`` is unknown, or ``default`` does not fit it.

        Examples:
            >>> self.add_state("hits", 0, "sum")
            >>> self.add_state("scores", [], "cat")
        """
        specs = self.__dict__.setdefault("_state_specs", {})
        if name in specs or hasattr(self, name):
            raise ValueError(f"state {name!r} is already defined")
        if reduce == "sum":
            if isinstance(default, bool) or not isinstance(default, (int, float)):
                raise ValueError(f"state {name!r}: 'sum' needs an int or float")
        elif reduce == "cat":
            if not isinstance(default, list):
                raise ValueError(f"state {name!r}: 'cat' needs a list")
        else:
            raise ValueError(f"state {name!r}: reduce must be 'sum' or 'cat'")
        specs[name] = (copy.deepcopy(default), reduce)
        setattr(self, name, copy.deepcopy(default))

    def reset(self) -> None:
        """Set every registered state back to a copy of its default.

        Examples:
            >>> metric.reset()
        """
        for name, (default, _) in self.__dict__.get("_state_specs", {}).items():
            setattr(self, name, copy.deepcopy(default))

    def state(self) -> Dict[str, StateValue]:
        """Return a deep copy of every registered state, by name.

        Returns:
            Plain Python values; ``{}`` when nothing is registered.

        Examples:
            >>> metric.state()
            {'hits': 0, 'scores': []}
        """
        specs = self.__dict__.get("_state_specs", {})
        return {name: copy.deepcopy(getattr(self, name)) for name in specs}

    def merge_states(
        self, states: Sequence[Mapping[str, StateValue]]
    ) -> Dict[str, StateValue]:
        """Combine per-rank state dicts by each state's reduce rule.

        Pure: ``self`` is not changed. ``"sum"`` states are added and
        ``"cat"`` states are concatenated in the order of ``states``.
        Gathering the dicts across ranks is the caller's job.

        Args:
            states: One :meth:`state` dict per rank.

        Returns:
            The merged dict; the defaults when ``states`` is empty.

        Raises:
            ValueError: A dict lacks a registered name or has another one.

        Examples:
            >>> metric.merge_states([{"hits": 2}, {"hits": 3}])
            {'hits': 5}
        """
        specs = self.__dict__.get("_state_specs", {})
        if not states:
            return {name: copy.deepcopy(d) for name, (d, _) in specs.items()}
        for one in states:
            self._check_state_names(one)
        merged: Dict[str, StateValue] = {}
        for name, (_, reduce) in specs.items():
            if reduce == "sum":
                merged[name] = sum(one[name] for one in states)
            else:
                merged[name] = [x for one in states for x in copy.deepcopy(one[name])]
        return merged

    def load_state(self, state: Mapping[str, StateValue]) -> None:
        """Set every registered state from ``state``, such as a merged one.

        Args:
            state: A dict with exactly the registered names.

        Raises:
            ValueError: ``state`` lacks a registered name or has another one.

        Examples:
            >>> metric.load_state(metric.merge_states(gathered))
            >>> metric.compute()
            {'WER': 4.3}
        """
        self._check_state_names(state)
        for name in self.__dict__.get("_state_specs", {}):
            setattr(self, name, copy.deepcopy(state[name]))

    def _check_state_names(self, state: Mapping[str, StateValue]) -> None:
        """Raise ``ValueError`` unless ``state`` has exactly the registered names."""
        names = set(self.__dict__.get("_state_specs", {}))
        if set(state) != names:
            raise ValueError(
                f"state names {sorted(state)} do not match the registered "
                f"{sorted(names)}"
            )

    @abstractmethod
    def update(self, **values: Sequence[Any]) -> None:
        """Add one batch to the state.

        Gets one list or tuple per declared input name, all of the same
        length; an optional input may be absent. The batch is checked
        against the declared inputs before this runs (see
        :meth:`__init_subclass__`). Adds counts or appends scores, never
        rates, and writes no file.

        Args:
            **values: One sequence per declared input name.

        Examples:
            >>> metric.update(ref=["a b"], hyp=["a c"])
        """

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Return the result from the state alone, with no file side effect.

        Returns:
            One value per declared output name.

        Raises:
            ValueError: There is nothing to score.

        Examples:
            >>> metric.compute()
            {'WER': 4.3}
        """

    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """Score one test set from its SCP files, as ``measure`` calls it.

        Resets the state, reads the inputs present in ``data``
        with :meth:`iter_inputs`, calls :meth:`update` once per
        :attr:`chunk_size` rows, calls :meth:`compute` once at the end,
        and then, only when ``output_dir`` is given, calls
        :meth:`write_artifacts` with ``output_dir / test_name``.

        Args:
            data (Dict[str, Path]): Mapping of input names to SCP
                files, such as ``ref.scp`` and ``hyp.scp``; the metrics
                config's ``inputs:`` binds each name to its source.
            test_name (str): Name of the test dataset (e.g., "test-other").
            output_dir (Optional[Path]): Root directory for artifacts;
                ``None`` writes nothing.

        Returns:
            Dict[str, float]: What :meth:`compute` returns.

        Examples:
            >>> metric({"ref": ref_scp, "hyp": hyp_scp}, "test-clean")
            {'WER': 4.3}
        """
        self.reset()
        self.collect_artifacts = output_dir is not None
        try:
            names = list(data)
            if names:
                chunk: Dict[str, List[str]] = {name: [] for name in names}
                for _, row in self.iter_inputs(data, *names):
                    for name in names:
                        chunk[name].append(row[name])
                    if len(chunk[names[0]]) == self.chunk_size:
                        self.update(**chunk)
                        chunk = {name: [] for name in names}
                if chunk[names[0]]:
                    self.update(**chunk)
            result = self.compute()
            if output_dir is not None:
                test_dir = Path(output_dir) / test_name
                test_dir.mkdir(parents=True, exist_ok=True)
                self.write_artifacts(test_dir)
            return result
        finally:
            self.collect_artifacts = False

    def write_artifacts(self, test_dir: Path) -> None:
        """Write files for one test set after :meth:`compute`; no-op by default.

        Runs only from :meth:`__call__`, and only when it has an
        ``output_dir``. Override it to write, for example, alignments kept
        in a ``"cat"`` state.

        Args:
            test_dir: ``output_dir / test_name``, already created.

        Examples:
            >>> metric.write_artifacts(Path("exp/metrics/test-clean"))
        """

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
