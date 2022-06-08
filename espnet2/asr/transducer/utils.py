"""Utility functions for Transducer models."""

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


def get_transducer_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
):
    """Get Transducer loss I/O.

    Args:
        labels: Label ID sequences. (B, L)
        encoder_out_lens: Encoder output lengths. (B,)
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.

    Return:
        decoder_in: Decoder inputs. (B, U)
        target: Target label ID sequences. (B, U)
        t_len: Time lengths. (B,)
        u_len: Label lengths. (B,)

    """
    device = labels.device

    labels_unpad = [y[y != ignore_id] for y in labels]
    blank = labels[0].new([blank_id])

    decoder_in = pad_list(
        [torch.cat([blank, label], dim=0) for label in labels_unpad], blank_id
    ).to(device)

    target = pad_list(labels_unpad, blank_id).type(torch.int32).to(device)

    if encoder_out_lens.dim() > 1:
        enc_mask = [m[m != 0] for m in encoder_out_lens]
        encoder_out_lens = list(map(int, [m.size(0) for m in enc_mask]))
    else:
        encoder_out_lens = list(map(int, encoder_out_lens))

    t_len = torch.IntTensor(encoder_out_lens).to(device)
    u_len = torch.IntTensor([y.size(0) for y in labels_unpad]).to(device)

    return decoder_in, target, t_len, u_len
