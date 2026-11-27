"""Core Atlas runtime components."""


class AtlasError(Exception):
    """Base class for Atlas CLI errors, raised for invalid input or study state.

    Examples:
        >>> raise AtlasError("atlas study failed")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.AtlasError: atlas study failed
    """


class InvalidAgentResponseError(AtlasError):
    """A coding agent's structured output stayed invalid after bounded repair.

    The scheduler catches this one separately: it records a failed result
    containing ``invalid_agent_response`` and a ``.stage_failure.json`` and
    pauses the node run instead of corrupting study state.

    Examples:
        >>> raise InvalidAgentResponseError("agent response is not valid JSON")
        Traceback (most recent call last):
        espnet3.bin.atlas.core.InvalidAgentResponseError: agent response is not ...
    """
