"""Atlas error hierarchy."""


class AtlasError(Exception):
    """Base class for all Atlas CLI exceptions.

    Examples:
        >>> raise AtlasError("atlas study failed")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.errors.AtlasError: atlas study failed
    """


class GraphRoutingError(AtlasError):
    """A graph node's ``on:`` routing table has no matching outcome.

    Raised when a stage's result does not match any key declared in its
    node's ``on:`` mapping (for example an unexpected ``result_status``).

    Examples:
        >>> raise GraphRoutingError("no route for outcome 'timeout'")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.errors.GraphRoutingError: no route for ...
    """


class StageLoadError(AtlasError):
    """A graph node's ``target`` class path cannot be imported.

    Raised when the dotted path in a node's ``target`` field does not
    resolve to an importable class.

    Examples:
        >>> raise StageLoadError("cannot import 'pkg.Missing'")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.errors.StageLoadError: cannot import ...
    """


class MetricNotFoundError(AtlasError):
    """A trial's configured metric key is missing from its result files.

    Raised when ``collect_result`` cannot find the configured extraction
    path/key in the trial's metrics output.

    Examples:
        >>> raise MetricNotFoundError("key 'wer' not found in metrics.json")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.errors.MetricNotFoundError: key 'wer' ...
    """


class ConfigValidationError(AtlasError):
    """An ``atlas.yaml`` study config is structurally invalid.

    Raised when a required field is missing or a field has an invalid
    value (for example an unknown ``agent.provider``).

    Examples:
        >>> raise ConfigValidationError("atlas.agent.provider must be 'codex' or 'claude'")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.errors.ConfigValidationError: atlas.agent ...
    """


class InvalidAgentResponseError(AtlasError):
    """A coding agent's structured output could not be parsed as JSON.

    Raised after bounded JSON repair retries fail, which pauses study
    scheduling for that node run instead of corrupting state.

    Examples:
        >>> raise InvalidAgentResponseError("agent response is not valid JSON")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.errors.InvalidAgentResponseError: agent ...
    """
