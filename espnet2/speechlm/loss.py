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
            self.loss = LigerFusedLinearCrossEntropyLoss(ignore_index=pad_id)
            self.fused = True
        except:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
            self.fused = False
            logging.warning(
                "liger_kernel is not available. Use Pytorch implementation. "
                "This will significantly increase memory peak. "
            )

        self.torch_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
        
    def __call__(self, hidden, targets):
        """
        hidden (torch.Tensor): hidden embeddings, typically output from transformer
          with lm_head bias. Size: (B, T, nq, D)
        targets (torch.Tensor): predicting target. Size (B, T, nq)
        """

        assert targets.size() == hidden.size()[:3]
        B, T, nq = targets.size()

        fused = self.fused and self.training

        hidden = hidden.view(B * T * nq, -1)
        targets = targets.contiguous().view(-1)

        # compute loss
        if fused: 
            logits = None
            loss = self.loss(self.lm_head.weight, hidden, targets)
        else:
            logits = self.lm_head(hidden)
            loss = self.torch_loss(logits, targets)

        weight = (targets != self.pad_id).float().sum().clone().detach()
        stats = {"loss": loss.clone().detach(), "weight": weight}
        
        # compute token accuracy
        if not fused:
            pred = logits.view(B, T, nq, -1).argmax(-1)
            targets = targets.view(B, T, nq)
            stats.update({"acc": self.compute_acc(pred, targets)})

            for n in range(nq):
                stats.update({
                    f"acc_layer{n}": self.compute_acc(pred[:, :, n], targets[:, :, n])
                })
        
        return loss, logits, stats, weight
    
    @torch.no_grad()
    def compute_acc(self, pred, targets):
        acc = torch.logical_and(
            pred == targets,
            targets != self.pad_id
        ).float().sum()
        count = (targets != self.pad_id).float().sum()
        return (acc / count).clone().detach()



