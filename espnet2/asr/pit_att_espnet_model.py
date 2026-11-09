"""ESPnetASRModel with a permutation-invariant attention loss."""

import itertools

import torch
from typeguard import typechecked

from espnet2.asr.decoder.linear_decoder import LinearDecoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.legacy.nets.pytorch_backend.nets_utils import th_accuracy
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos


class PITAttESPnetModel(ESPnetASRModel):
    """Encoder-decoder model with a permutation-invariant attention loss.

    The attention loss of each utterance is the minimum over all orders of its
    target tokens (one teacher-forced decoder pass per distinct order), so it
    suits short unordered targets such as language-ID sequences. Accuracy and
    error rates use the chosen order. Attention-only: CTC, transducer,
    LinearDecoder and lang_token_id are not supported.

    Args:
        max_pit_tokens: Maximum target length. Longer targets raise an error
            since the number of decoder passes grows factorially.
        **kwargs: Arguments of ESPnetASRModel.
    """

    @typechecked
    def __init__(self, *, max_pit_tokens: int = 3, **kwargs):
        """Reject unsupported options and build ESPnetASRModel."""
        if kwargs.get("ctc_weight", 0.5) != 0.0:
            raise ValueError("PITAttESPnetModel is attention-only: set ctc_weight=0")
        if kwargs.get("joint_network") is not None:
            raise ValueError("PITAttESPnetModel does not support a transducer")
        if isinstance(kwargs.get("decoder"), LinearDecoder):
            raise ValueError("PITAttESPnetModel does not support LinearDecoder")
        if kwargs.get("lang_token_id", -1) != -1:
            raise ValueError("PITAttESPnetModel does not support lang_token_id")
        super().__init__(**kwargs)
        self.max_pit_tokens = max_pit_tokens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        max_len = int(ys_pad_lens.max())
        if max_len > self.max_pit_tokens:
            raise ValueError(
                f"Target of length {max_len} exceeds max_pit_tokens="
                f"{self.max_pit_tokens}"
            )

        # 1. Loss for the given order
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_pad_lens + 1
        )
        best_loss = self._per_sample_att_loss(decoder_out, ys_out_pad)

        # 2. Keep, per utterance, the order with the lowest loss
        best_ys_pad = ys_pad
        for perm in itertools.permutations(range(max_len)):
            moved = [i for i, j in enumerate(perm) if i != j]
            if not moved:
                continue
            perm_ys_pad = ys_pad.clone()
            perm_ys_pad[:, :max_len] = ys_pad[:, list(perm)]
            # Reorder only within each utterance and skip orders that give
            # the same sequence (e.g. two identical tokens)
            valid = (ys_pad_lens > moved[-1]) & (perm_ys_pad != ys_pad).any(dim=1)
            if not bool(valid.any()):
                continue
            perm_ys_in_pad, perm_ys_out_pad = add_sos_eos(
                perm_ys_pad, self.sos, self.eos, self.ignore_id
            )
            perm_decoder_out, _ = self.decoder(
                encoder_out, encoder_out_lens, perm_ys_in_pad, ys_pad_lens + 1
            )
            perm_loss = self._per_sample_att_loss(perm_decoder_out, perm_ys_out_pad)
            better = valid & (perm_loss < best_loss)
            best_loss = torch.where(better, perm_loss, best_loss)
            decoder_out = torch.where(
                better[:, None, None], perm_decoder_out, decoder_out
            )
            best_ys_pad = torch.where(better[:, None], perm_ys_pad, best_ys_pad)
            ys_out_pad = torch.where(better[:, None], perm_ys_out_pad, ys_out_pad)

        # 3. Reduce as LabelSmoothingLoss does
        if self.criterion_att.normalize_length:
            denom = (ys_out_pad != self.ignore_id).sum().clamp(min=1)
            loss_att = best_loss.sum() / denom
        else:
            loss_att = best_loss.mean()
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), best_ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _per_sample_att_loss(
        self, decoder_out: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return the unreduced LabelSmoothingLoss of each utterance."""
        criterion = self.criterion_att
        batch_size, maxlen, vocab_size = decoder_out.size()
        x = decoder_out.reshape(-1, vocab_size)
        target = target.reshape(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(criterion.smoothing / (criterion.size - 1))
            ignore = target == criterion.padding_idx
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), criterion.confidence)
        kl = criterion.criterion(torch.log_softmax(x, dim=1), true_dist)
        token_loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum(dim=1)
        return token_loss.reshape(batch_size, maxlen).sum(dim=1)
