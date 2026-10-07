"""Base interface for dataset builders."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, ClassVar, Dict, Optional, Tuple

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import check_fields, check_manifests


class DatasetBuilder(ABC):
    """Interface for recipe-local dataset preparation helpers.

    This abstract class defines the contract used by the
    ``create_dataset`` stage when a dataset source such as
    ``data_src: mini_an4/esp2_asr`` appears in a training or inference config.
    Concrete builders separate source acquisition from task-specific artifact
    generation so the system can skip work precisely:

    1. ``is_source_prepared(**kwargs)`` checks whether the raw source tree is
       already available.
    2. ``prepare_source(**kwargs)`` downloads, copies, validates, or extracts
       that raw source tree if step 1 returns ``False``.
    3. ``is_built(**kwargs)`` checks whether task-ready outputs already exist.
    4. ``build(**kwargs)`` creates those outputs from the prepared source tree
       if step 3 returns ``False``.

    Concrete implementations are expected to be idempotent. Re-running
    ``create_dataset`` should not rewrite the same source tree or manifests
    unnecessarily when the expected outputs are already present.

    A builder that writes its own espnet3 manifest declares
    ``manifest_columns`` (and overrides ``built_manifests``) - a class
    attribute for a fixed set of columns, or set on ``self`` (in its own
    ``__init__``) for one that depends on its own configuration. ``build()``
    then checks each returned manifest's first row against the declaration
    itself; there is no undeclared fallback once ``built_manifests()``
    actually returns something. A builder with no manifest of its own (it
    reads its source corpus directly) needs neither.

    Notes:
        ``espnet3.systems.base.system.BaseSystem.create_dataset()`` instantiates
        a builder and calls these methods in order. Keep
        ``is_source_prepared`` and ``is_built`` as cheap filesystem checks, and
        reserve heavier work for ``prepare_source`` and ``build``.

    Examples:
        A recipe config can trigger a builder through a dataset reference:
        ```yaml
        dataset:
          train:
            - data_src: mini_an4/esp2_asr
              data_src_args:
                split: train
        ```

        The system then follows the builder lifecycle:
        ```python
        builder = DatasetBuilderSubclass()
        if not builder.is_source_prepared(recipe_dir="."):
            builder.prepare_source(recipe_dir=".")
        if not builder.is_built(recipe_dir="."):
            builder.build(recipe_dir=".")
        ```
    """

    @abstractmethod
    def is_source_prepared(self, **kwargs) -> bool:
        """Check whether the raw source artifacts are already available.

        This method should answer only whether the source side is ready for a
        later ``build()`` call. Typical checks include the existence of an
        extracted corpus directory, a verified archive, or a copied upstream
        dataset tree.

        Args:
            **kwargs: Builder-specific lookup parameters, typically including
                values such as ``recipe_dir`` or dataset-scoped options from the
                config entry.

        Returns:
            bool: ``True`` when the raw source assets are already present and
            usable, otherwise ``False``.

        Raises:
            None: Implementations should prefer returning ``False`` for simple
                absence checks. Exceptional states such as corrupt metadata may
                raise implementation-specific errors.

        Notes:
            Keep this check cheap. It may be called frequently from stage logic
            to determine whether heavier preparation work can be skipped.

        Examples:
            ```python
            if builder.is_source_prepared(recipe_dir="."):
                print("source already available")
            ```

            A task-scoped dataset may treat an extracted directory as the source
            readiness marker:
            ```python
            ready = (source_root / "an4").is_dir()
            ```
        """

    @abstractmethod
    def prepare_source(self, **kwargs) -> None:
        """Materialize the raw source artifacts needed by the dataset.

        Implementations typically download archives, copy data from another
        recipe, verify checksums, or extract compressed files into a stable
        source directory.

        Args:
            **kwargs: Builder-specific preparation arguments such as
                ``recipe_dir`` or archive locations resolved from config.

        Returns:
            None: Source preparation is communicated through filesystem side
            effects rather than return values.

        Notes:
            This method should leave the dataset in a state where
            ``is_source_prepared(**kwargs)`` becomes ``True`` immediately after
            successful completion.

        Examples:
            ```python
            builder.prepare_source(recipe_dir="egs3/mini_an4/esp2_asr")
            ```

            An implementation may extract a bundled archive into ``source/``:
            ```python
            prepare_source(source_dir=source_root, archive_path=archive_path)
            ```
        """

    @abstractmethod
    def is_built(self, **kwargs) -> bool:
        """Check whether task-ready dataset artifacts already exist.

        This method should report readiness of the outputs consumed by ESPnet3
        components, such as manifests, converted audio files, feature metadata,
        or recipe-specific index files.

        Args:
            **kwargs: Builder-specific lookup parameters, typically including
                ``recipe_dir`` and any config-derived dataset options needed to
                locate built artifacts.

        Returns:
            bool: ``True`` when the task-ready dataset outputs are complete
            enough to skip ``build()``, otherwise ``False``.

        Raises:
            None: Implementations should normally return ``False`` when outputs
                are absent. Exceptions are appropriate only for clearly invalid
                states such as contradictory configuration.

        Notes:
            Keep this method cheap and deterministic. It is intended for stage
            planning, not for performing repairs or partial generation.

        Examples:
            ```python
            if not builder.is_built(recipe_dir="."):
                builder.build(recipe_dir=".")
            ```

            A manifest-based recipe might check for a small fixed set of files:
            ```python
            ready = all((root / relpath).is_file() for relpath in required_files)
            ```
        """

    @abstractmethod
    def build(self, **kwargs) -> None:
        """Build task-ready artifacts from the prepared source artifacts.

        This method transforms the prepared source tree into the exact dataset
        outputs expected by the task, such as manifest TSVs, normalized text,
        converted audio, or metadata consumed by training and inference.

        Args:
            **kwargs: Builder-specific build arguments, commonly including
                ``recipe_dir`` and dataset entry options such as ``split``.

        Returns:
            None: Build results are written to the filesystem.

        Notes:
            ``build()`` should assume the source side is already available. It
            is normally called only after ``is_source_prepared`` and
            ``prepare_source`` have been handled by the system.

        Examples:
            ```python
            builder.build(recipe_dir="egs3/mini_an4/esp2_asr")
            ```

            A task-scoped builder can consume shared source assets and emit
            recipe-local manifests:
            ```python
            build_dataset(dataset_dir=dataset_root, source_dir=source_root)
            ```
        """

    #: The manifest's columns, in file order. ``None`` (the default) means
    #: the manifest is not checked, which is only valid when
    #: ``built_manifests`` also stays at its default (an empty dict); a
    #: builder that writes a manifest must declare this.
    manifest_columns: ClassVar[Optional[Tuple[Field, ...]]] = None
    #: Whether each manifest file has a header row to skip.
    manifest_header: ClassVar[bool] = False

    def built_manifests(self, **kwargs) -> Dict[str, Path]:
        """Return a split-to-manifest-path mapping for what ``build()`` wrote.

        The default returns ``{}``, which skips the manifest check (declare
        ``manifest_columns`` and override this to opt in).

        Args:
            **kwargs: The same arguments ``build()`` was called with.

        Returns:
            A mapping from split name to the manifest path ``build()``
            wrote for it.

        Examples:
            >>> class MyBuilder(DatasetBuilder):
            ...     manifest_columns = (Field("utt_id", "text"), Field("text", "text"))
            ...     def is_source_prepared(self, **kwargs): return True
            ...     def prepare_source(self, **kwargs): pass
            ...     def is_built(self, **kwargs): return True
            ...     def build(self, recipe_dir, **kwargs): pass
            ...     def built_manifests(self, recipe_dir, **kwargs):
            ...         return {"train": Path(recipe_dir) / "train.tsv"}
        """
        return {}

    def __init_subclass__(cls, **kwargs):
        """Validate class-level ``manifest_columns`` and check after ``build``."""
        super().__init_subclass__(**kwargs)
        if cls.manifest_columns is not None:
            check_fields(cls, "manifest_columns")
        if not getattr(cls.build, "__wrapped_contract__", False):
            cls.build = _check_manifests_after(cls.build)


def _as_kwargs(func: Callable, self, args: tuple, kwargs: dict) -> dict:
    """Flatten a bound call's positional and ``**kwargs`` arguments to kwargs.

    ``build()`` may take some arguments positionally (``recipe_dir`` by
    convention); ``built_manifests()`` needs the same ones by name.
    """
    sig = inspect.signature(func)
    bound = sig.bind(self, *args, **kwargs)
    flat: dict = {}
    for name, value in bound.arguments.items():
        if name == "self":
            continue
        if sig.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
            flat.update(value)
        else:
            flat[name] = value
    return flat


def _check_manifests_after(build: Callable) -> Callable:
    """Wrap ``build`` to check its written manifests against the declaration."""

    @functools.wraps(build)
    def wrapper(self, *args, **kwargs):
        result = build(self, *args, **kwargs)
        check_manifests(self, **_as_kwargs(build, self, args, kwargs))
        return result

    wrapper.__wrapped_contract__ = True
    return wrapper
