"""FalAR dataset module."""

__all__ = ["Dataset", "DatasetBuilder"]


def __getattr__(name):
    """Load dataset exports only when the runner asks for them."""
    if name == "DatasetBuilder":
        from egs3.falar.asr.dataset.builder import FalarBuilder

        return FalarBuilder
    if name == "Dataset":
        from egs3.falar.asr.dataset.dataset import FalarDataset

        return FalarDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
