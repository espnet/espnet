"""Helpers for the optional kaldiio dependency."""

try:
    import kaldiio
except ImportError:
    kaldiio = None


KALDIIO_INSTALL_MESSAGE = (
    "kaldiio is not installed. "
    'Please install the optional dependency with: pip install "espnet[kaldi]"'
)


def require_kaldiio():
    if kaldiio is None:
        raise ImportError(KALDIIO_INSTALL_MESSAGE)
    return kaldiio
