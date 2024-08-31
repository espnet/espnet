import torch
import logging

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, lm_head, pad_id=0):
        """ Compute CrossEntropy loss for multi-stream LM using liger fused triton kernel """
        super(FusedLinearCrossEntropyLoss, self).__init__()
        self.lm_head = lm_head
        self.pad_id = pad_id

        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            self.loss = LigerFusedLinearCrossEntropyLoss()
            self.fused = True
        except:
            self.loss = torch.nn.CrossEntropyLoss()
            self.fused = False
            logging.warning("liger_kernel is not available. Use Pytorch implementation")

        self.torch_loss = torch.nn.CrossEntropyLoss()
        
    def __call__(self, hidden, targets):
        """
        hidden (torch.Tensor): hidden embeddings, typically output from transformer
          with lm_head bias. Size: (B, T, nq, D)
        targets (torch.Tensor): predicting target. Size (B, T, nq)
        """

        assert targets.size() == hidden.size()[:3]
        B, T, nq = targets.size()

        fused = self.fused and self.training

        # select items that are not padding.
        padding_mask = targets != self.pad_id
        hidden = hidden[padding_mask]
        targets = targets[padding_mask]

        # compute loss
        if fused: 
            logits = None
            loss = self.loss(self.lm_head.weight, hidden, targets)
        else:
            logits = self.lm_head(hidden)
            loss = self.torch_loss(logits, targets)
        weight = targets.numel()
        stats = {"loss": loss.clone().detach()}
        
        # compute token accuracy
        if not fused:
            layer_idx = torch.arange(nq, device=hidden.device).tile(B, T, 1)
            layer_idx = layer_idx[padding_mask]

            for idx in range(nq):
                acc = torch.logical_and(
                    logits.argmax(-1) == targets,
                    layer_idx == idx
                ).float().sum()
                acc = acc / (layer_idx == idx).float().sum()
                stats[f"acc_layer{idx}"] = acc.clone().detach()

        return loss, logits, stats, weight



