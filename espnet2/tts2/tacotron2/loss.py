import torch

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class Tacotron2LossDiscrete(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self,
        idim,
        vocab_size,
        use_masking=True,
        use_weighted_masking=False,
        bce_pos_weight=20.0,
    ):
        """Initialize Tactoron2 loss module.

        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token.

        """
        super(Tacotron2LossDiscrete, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        self.linear = torch.nn.Linear(idim, vocab_size, bias=False)
        self.idim = idim

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor(bce_pos_weight)
        )


        # NOTE(kan-bayashi): register pre hook function for the compatibility
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks.unsqueeze(-1)).view(
                -1, self.idim
            )
            before_outs = before_outs.masked_select(masks.unsqueeze(-1)).view(
                -1, self.idim
            )
            labels = labels.masked_select(masks)
            logits = logits.masked_select(masks)

        # calculate loss
        before_outs = self.linear(before_outs)
        after_outs = self.linear(after_outs)
        ce_loss = self.ce_criterion(after_outs, ys) + self.ce_criterion(before_outs, ys)
        bce_loss = self.bce_criterion(logits, labels)

        before_acc = torch.eq(before_outs.argmax(dim=-1), ys).int().sum() / len(ys)
        after_acc = torch.eq(after_outs.argmax(dim=-1), ys).int().sum() / len(ys)
        bce_acc = torch.ge(torch.sigmoid(logits), 0.5).eq(labels.bool()).int().sum() / len(labels)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))
            logit_weights = weights.div(ys.size(0))

            # apply weight
            ce_loss = ce_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = (
                bce_loss.mul(logit_weights.squeeze(-1))
                .masked_select(masks.squeeze(-1))
                .sum()
            )
        
        stats = {
            'ce_loss': ce_loss.item(),
            'bce_loss': bce_loss.item(),
            'before_acc': before_acc.item(),
            'after_acc': after_acc.item(),
            'bce_acc': bce_acc.item()
        }

        return ce_loss, bce_loss, stats

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Apply pre hook function before loading state dict.

        From v.0.6.1 `bce_criterion.pos_weight` param is registered as a parameter but
        old models do not include it and as a result, it causes missing key error when
        loading old model parameter. This function solve the issue by adding param in
        state dict before loading as a pre hook function
        of the `load_state_dict` method.

        """
        key = prefix + "bce_criterion.pos_weight"
        if key not in state_dict:
            state_dict[key] = self.bce_criterion.pos_weight
