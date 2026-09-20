"""Metrics scored on one hypothesis/reference pair at a time."""

from splet.utterance_metrics.error_rate import (  # noqa: F401
    error_rate_metric,
    error_rate_setup,
)

__all__ = ["error_rate_metric", "error_rate_setup"]
