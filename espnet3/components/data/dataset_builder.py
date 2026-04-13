"""Base interface for dataset builders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetBuilder(ABC):
    """Interface for recipe-local dataset preparation helpers.

    This abstract class defines the contract used by the
    ``create_dataset`` stage when a dataset source such as
    ``data_src: mini_an4/asr`` appears in a training or inference config.
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
            - data_src: mini_an4/asr
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
            builder.prepare_source(recipe_dir="egs3/mini_an4/asr")
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
            builder.build(recipe_dir="egs3/mini_an4/asr")
            ```

            A task-scoped builder can consume shared source assets and emit
            recipe-local manifests:
            ```python
            build_dataset(dataset_dir=dataset_root, source_dir=source_root)
            ```
        """
