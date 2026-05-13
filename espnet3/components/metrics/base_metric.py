"""Base metric interfaces for ESPnet3."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Tuple


class BaseMetric(ABC):
    """Base class for metrics that consume inference output paths."""

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
                          _target_: espnet3.systems.asr.metrics.wer.WER
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
