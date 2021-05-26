"""Create mask for subsequent steps."""


def make_history_mask(xp, block):
    """Prepare the history mask.

    Args:
        block (ndarray): Block with dimensions: (B x S).
    Returns:
        ndarray, np.ndarray: History mask with dimensions (B, S, S).

    """
    batch, length = block.shape
    arange = xp.arange(length)
    history_mask = (arange[None] <= arange[:, None])[None,]
    history_mask = xp.broadcast_to(history_mask, (batch, length, length))
    return history_mask
