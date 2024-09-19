import torch

from espnet2.speechlm.net_utils import length_mask


class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, lm_head, pad_id=0, prefix_lm=True, chunk_size=32768):
        """
        Compute CrossEntropy loss for multi-stream LM using either:
        (1) liger fused triton kernel
            slower but much less memory consumption
        (2) torch implementation
            normal speed but large memory consumption

        """
        super(FusedLinearCrossEntropyLoss, self).__init__()
        self.lm_head = lm_head
        self.pad_id = pad_id
        self.prefix_lm = prefix_lm
        self.chunk_size = chunk_size

        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            self.loss = LigerFusedLinearCrossEntropyLoss()
            self.fused = True
        except:
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
            self.fused = False

        self.torch_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def __call__(self, hidden, targets, prefix_len=None):
        """
        hidden (torch.Tensor): hidden embeddings, typically output from transformer
          with lm_head bias. Size: (B, T, nq, D)
        targets (torch.Tensor): predicting target. Size (B, T, nq)
        """

        assert targets.size() == hidden.size()[:3]
        B, T, nq = targets.size()

        # fused = self.fused and self.training
        fused = False

        # select items that are not padding. This mask select is fast and will save
        # the computing on padding tokens (very massive in multi-stream case).
        padding_mask = targets != self.pad_id

        if self.prefix_lm:
            if prefix_len is None:
                raise ValueError("No prefix_len to compute prefix_lm loss")
            prefix_mask = ~length_mask(prefix_len, maxlen=targets.size(1)).unsqueeze(2)
            mask = torch.logical_and(padding_mask, prefix_mask)

        else:
            mask = padding_mask

        hidden = hidden[mask]
        targets = targets[mask]

        # compute loss
        if fused:
            logits = None
            loss = self.loss(self.lm_head.weight, hidden, targets)
        else:
            # chunk-by-chunk CE loss to avoid memory peak
            chunk_id, logits, loss = 0, [], []
            while chunk_id * self.chunk_size < len(hidden):
                start = chunk_id * self.chunk_size
                end = min((chunk_id + 1) * self.chunk_size, len(hidden))
                this_logits = self.lm_head(hidden[start:end])
                this_targets = targets[start:end]
                this_loss = self.torch_loss(this_logits, this_targets)
                logits.append(this_logits)
                loss.append(this_loss)
                chunk_id += 1
            loss = torch.cat(loss).mean()
        weight = float(targets.numel())
        stats = {"loss": loss.clone().detach(), "weight": weight}

        # compute token accuracy
        if not fused and not self.training:
            logits = torch.cat(logits, dim=0)
            layer_idx = torch.arange(nq, device=hidden.device).tile(B, T, 1)
            layer_idx = layer_idx[padding_mask]

            for idx in range(nq):
                acc = (
                    torch.logical_and(logits.argmax(-1) == targets, layer_idx == idx)
                    .float()
                    .sum()
                )
                acc = acc / (layer_idx == idx).float().sum()
                stats[f"acc_layer{idx}"] = acc.clone().detach()

            acc = (logits.argmax(-1) == targets).float().sum()
            acc = acc / targets.numel()
            stats["acc"] = acc.clone().detach()

        return loss, logits, stats, weight


if __name__ == "__main__":
    hidden = torch.randn((1, 7, 2, 512)).float().cuda() * 100
    target = torch.randint(0, 9, (1, 7, 2)).long().cuda()
    print("target: ", target)
    prefix_len = torch.Tensor([6]).long().cuda()
    linear = torch.nn.Linear(512, 9).cuda()

    liger_loss = FusedLinearCrossEntropyLoss(
        linear, pad_id=80000, prefix_lm=True
    ).cuda()
    # torch_loss = torch.nn.CrossEntropyLoss(ignore_index=80000)

    loss_liger, _, _, _ = liger_loss(hidden, target, prefix_len)
    # loss_torch = torch_loss(linear(hidden).view(-1, 70032), target.view(-1))

    # print('loss_liger', 'loss_torch', loss_liger, loss_torch)
