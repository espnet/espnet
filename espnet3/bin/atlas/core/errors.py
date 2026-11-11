"""Atlas error hierarchy."""


class AtlasError(Exception):
    """Base class for Atlas exceptions."""


class GraphRoutingError(AtlasError):
    """Raised when graph routing is invalid."""


class StageLoadError(AtlasError):
    """Raised when a stage class cannot be imported."""


class MetricNotFoundError(AtlasError):
    """Raised when a configured metric cannot be extracted."""


class ConfigValidationError(AtlasError):
    """Raised when an Atlas config is invalid."""


class InvalidAgentResponseError(AtlasError):
    """Raised after bounded JSON repairs fail, pausing study scheduling."""
