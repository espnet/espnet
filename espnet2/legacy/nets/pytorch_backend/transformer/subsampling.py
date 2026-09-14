#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import torch

from espnet2.legacy.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv1dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling2) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling3) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv2dSubsampling2) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


def _conv_out_length(length, kernel_size, stride, padding=0, dilation=1):
    """Time-axis output length of a convolution (PyTorch Conv1d/Conv2d formula)."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _subsample_mask(x_mask, out_time, convs):
    """Rebuild the pad mask from each utterance's own input length.

    ``convs`` is a sequence of ``(kernel_size, stride)`` matching the time-axis
    convolutions of the module. Slicing the padded batch
    (``x_mask[:, :, :-k:s]``) trims relative to ``max(ilens)``, so a short
    utterance reports a different ``olens`` depending on its batch mates
    (issue 6600).

    """
    if x_mask is None:
        return None
    olens = x_mask.squeeze(1).sum(dim=-1)
    for kernel_size, stride in convs:
        olens = _conv_out_length(olens, kernel_size, stride)
    olens = olens.clamp(min=0, max=out_time)
    frames = torch.arange(out_time, device=x_mask.device)
    mask = frames.unsqueeze(0) < olens.unsqueeze(1)
    return mask.to(dtype=x_mask.dtype).unsqueeze(1)


def _receptive_field_and_stride(convs):
    """Time-axis receptive field and total stride of stacked (kernel, stride) convs."""
    receptive_field, stride = 1, 1
    for kernel_size, s in convs:
        receptive_field += (kernel_size - 1) * stride
        stride *= s
    return receptive_field, stride


def _conv2d_project(module, x, convs):
    """Apply ``module.conv`` and ``module.out`` to ``x`` (#batch, time, idim).

    At inference a batch is usually padded with one repeated frame: zeros that
    the normalization turns into a fixed log-mel row, or, for a model such as
    OWSM that pads every input to 30 s, a long run of them. A convolution over
    identical frames gives identical outputs, so when every utterance of the
    batch ends in such a run, the convolutions are computed only up to the
    first output frame whose receptive field lies inside the run, and that
    frame is repeated for the rest. Frames whose receptive field touches real
    content are computed as usual, so the result is that of the full
    computation. The check is one comparison of every frame with the last one.
    The full computation is used in training mode, when the tail is not
    constant, and when ``module.skip_constant_tail`` is set to False.

    ``convs`` lists the time-axis ``(kernel_size, stride)`` of ``module.conv``.
    """
    if not module.training and getattr(module, "skip_constant_tail", True):
        time = x.size(1)
        same_as_last = (x == x[:, -1:, :]).all(dim=2)  # (#batch, time)
        trailing = same_as_last.flip(1).to(torch.int64).cumprod(1).sum(1)
        first_pad = int((time - trailing).max())
        receptive_field, stride = _receptive_field_and_stride(convs)
        t_crop = min(time, first_pad + receptive_field + stride)
        if t_crop < time:
            out_time = time
            for kernel_size, s in convs:
                out_time = _conv_out_length(out_time, kernel_size, s)
            y = module.conv(x[:, :t_crop].unsqueeze(1))
            b, c, t, f = y.size()
            y = module.out(y.transpose(1, 2).contiguous().view(b, t, c * f))
            if t < out_time:
                y = torch.cat([y, y[:, -1:, :].expand(b, out_time - t, -1)], dim=1)
            return y
    x = x.unsqueeze(1)  # (b, c, t, f)
    x = module.conv(x)
    b, c, t, f = x.size()
    return module.out(x.transpose(1, 2).contiguous().view(b, t, c * f))


def _upgrade_legacy_subsampling_state_dict(state_dict, prefix):
    """Remap legacy nn.Sequential keys for subsampling modules."""
    w_new = prefix + "out.weight"
    b_new = prefix + "out.bias"
    w_old = prefix + "out.0.weight"
    b_old = prefix + "out.0.bias"

    if w_new not in state_dict and w_old in state_dict:
        state_dict[w_new] = state_dict.pop(w_old)
    elif w_new in state_dict and w_old in state_dict:
        state_dict.pop(w_old)

    if b_new not in state_dict and b_old in state_dict:
        state_dict[b_new] = state_dict.pop(b_old)
    elif b_new in state_dict and b_old in state_dict:
        state_dict.pop(b_old)

    old_pos_prefix = prefix + "out.1."
    new_pos_prefix = prefix + "pos_enc."
    for k in list(state_dict.keys()):
        if not k.startswith(old_pos_prefix):
            continue
        new_k = new_pos_prefix + k[len(old_pos_prefix) :]
        if new_k not in state_dict:
            state_dict[new_k] = state_dict[k]
        state_dict.pop(k, None)


class Conv1dSubsampling1(torch.nn.Module):
    """Convolutional 1D subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling1 object."""
        super(Conv1dSubsampling1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim, odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 1), (3, 1)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.pos_enc


class Conv1dSubsampling2(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling2 object."""
        super(Conv1dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim, odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 1), (3, 2)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.pos_enc


class Conv1dSubsampling3(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/3 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling3 object."""
        super(Conv1dSubsampling3, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim, odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 1), (5, 3)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.pos_enc


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = _conv2d_project(self, x, ((3, 2), (3, 2)))
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 2), (3, 2)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask

    # def __getitem__(self, key):
    #     """Get item.

    #     When reset_parameters() is called, if use_scaled_pos_enc is used,
    #         return the positioning encoding.

    #     """
    #     if key != -1:
    #         raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
    #     return self.out[key]


class Conv2dSubsampling1(torch.nn.Module):
    """Similar to Conv2dSubsampling module, but without any subsampling performed.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling1 object."""
        super(Conv2dSubsampling1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (idim - 4), odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Pass x through 2 Conv2d layers without subsampling.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim).
                where time' = time - 4.
            torch.Tensor: Subsampled mask (#batch, 1, time').
                where time' = time - 4.

        """
        x = _conv2d_project(self, x, ((3, 1), (3, 1)))
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 1), (3, 1)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.pos_enc


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = _conv2d_project(self, x, ((3, 2), (3, 1)))
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 2), (3, 1)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.pos_enc


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = _conv2d_project(self, x, ((3, 2), (5, 3)))
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 2), (5, 3)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _upgrade_legacy_subsampling_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = _conv2d_project(self, x, ((3, 2), (3, 2), (3, 2)))
        if x_mask is not None:
            x_mask = _subsample_mask(x_mask, x.size(1), ((3, 2), (3, 2), (3, 2)))

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask
