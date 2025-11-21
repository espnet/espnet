"""Transform Interface module."""


# TODO(karita): add this to all the transform impl.
class TransformInterface:
    """Transform Interface."""

    def __call__(self, x):
        """Process call function."""
        raise NotImplementedError("__call__ method is not implemented")

    @classmethod
    def add_arguments(cls, parser):
        """Add arguments to parser."""
        return parser

    def __repr__(self):
        """Return string with details of class."""
        return self.__class__.__name__ + "()"


class Identity(TransformInterface):
    """Identity Function."""

    def __call__(self, x):
        """Return same value."""
        return x
