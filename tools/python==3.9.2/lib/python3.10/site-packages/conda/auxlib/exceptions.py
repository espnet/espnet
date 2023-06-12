from logging import getLogger

log = getLogger(__name__)


def Raise(exception):  # NOQA
    raise exception


class AuxlibError:
    """Mixin to identify exceptions associated with the auxlib package."""


class AuthenticationError(AuxlibError, ValueError):
    pass


class NotFoundError(AuxlibError, KeyError):
    pass


class InitializationError(AuxlibError, EnvironmentError):
    pass


class SenderError(AuxlibError, IOError):
    pass


class AssignmentError(AuxlibError, AttributeError):
    pass


class ValidationError(AuxlibError, TypeError):
    def __init__(self, key, value=None, valid_types=None, msg=None):
        self.__cause__ = None  # in python3 don't chain ValidationError exceptions
        if msg is not None:
            super().__init__(msg)
        elif value is None:
            super().__init__("Value for {} cannot be None." "".format(key))
        elif valid_types is None:
            super().__init__("Invalid value {} for {}" "".format(value, key))
        else:
            super().__init__(
                "{} must be of type {}, not {}" "".format(key, valid_types, repr(value))
            )


class ThisShouldNeverHappenError(AuxlibError, AttributeError):
    pass
