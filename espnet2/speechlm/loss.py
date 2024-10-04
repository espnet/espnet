import torch
import logging
from espnet2.speechlm.net_utils import length_mask


class SpeechLMCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        pad,
        vocab_size,
        token_bias,
        loss_region,
        modality_weights,
    ):
        """
        Compute the CrossEntropy for SpeechLM. The main motivation of this module is to save computing.
        From the second layer and then on, the target codes can only be the codec codes or paddings, 
        which helps us to shrink the vocabulary during the loss computing. 
        """
        super().__init__()

        self.pad = pad
        self.loss_region = loss_region

        if "codec" in token_bias:
            self.aux_start, self.aux_end = token_bias["codec"]
        else:
            self.aux_start, self.aux_end = 0, 0

        # prepare the weight for first-layer
        weight = torch.ones(vocab_size).float()
        for modality_name, modality_weight in modality_weights.items():

            if modality_name not in token_bias:
                raise ValueError(f"modality {modality_name} is invalid.")
            
            start, end = token_bias[modality_name]
            del token_bias[modality_name]
            weight[start: end] = modality_weight

        for modality in token_bias.keys():
            logging.warning(f"weight for modality {modality} not specified. Set to 1.0")
        
        self.weight = weight
        
        if self.aux_start > 0 and self.aux_end > 0:
            aux_weight = weight[self.aux_start: self.aux_end]
            self.aux_weight = aux_weight
        else:
            self.aux_weight = None
    
    def forward(
        self, 
        logits: torch.Tensor,
        targets: torch.Tensor, 
        prefix_len: torch.Tensor,
        seq_len: torch.Tensor,
    ):
        # NOTE(Jinchuan): keep the weight on the correct device in the first forward.
        # We don't want to keep the weights registered as model parameters as they 
        # should always be specified by parameters.
        device, dtype = logits[0].device, logits[0].dtype
        self.weight = self.weight.to(device).to(dtype)
        if self.aux_weight is not None: 
            self.aux_weight = self.aux_weight.to(device).to(dtype)

        logits, aux_logits = logits
        assert logits.dim() == 4 and logits.size(2) == 1
        B, T, _, _ = logits.size()
        
        if aux_logits is not None:
            assert logits.dim() == 4
            assert logits.size()[:2] == aux_logits.size()[:2]
            assert logits.size(2) + aux_logits.size(2) == targets.size(2)
        
        assert prefix_len.size() == seq_len.size()
        assert torch.all(seq_len > prefix_len)
        assert prefix_len.dim() == 1

        # element-wise loss
        ce_loss = torch.nn.functional.cross_entropy(
            logits.flatten(end_dim=2),
            targets[:, :, :1].flatten(),
            self.weight,
            ignore_index=self.pad,
            reduction='none',
        ).view(B, T, 1)

        if aux_logits is not None:
            aux_mask = targets[:, :, 1:] != self.pad
            aux_targets = torch.clip(
                targets[:, :, 1:] - self.aux_start,
                min=0,
                max=self.aux_end - self.aux_start,
            )
            aux_ce_loss = torch.nn.functional.cross_entropy(
                aux_logits.flatten(end_dim=2),
                aux_targets.flatten(),
                weight=self.aux_weight,
                ignore_index=self.pad - self.aux_start,
                reduction='none',
            ).view(B, T, -1)
            aux_ce_loss = torch.where(aux_mask, aux_ce_loss, 0)

            ce_loss = torch.cat([ce_loss, aux_ce_loss], dim=2)
        
        mask = targets != self.pad
        weight = seq_len.sum().float()
        if self.loss_region == "target":
            prefix_mask = ~length_mask(prefix_len, maxlen=targets.size(1)).unsqueeze(2)
            mask = torch.logical_and(mask, prefix_mask)
            weight = (seq_len - prefix_len).sum().float()
        
        ce_loss = torch.where(mask, ce_loss, 0.0)
        ce_loss = ce_loss.sum() / weight
        stats = {"ce_loss": ce_loss.clone().detach(), "weight": weight.clone().detach()}

        if not self.training:
            acc = torch.logical_and(
                logits.argmax(-1) == targets[:, :, :1],
                mask[:, :, :1]
            )
            if aux_logits is not None:
                aux_acc = torch.logical_and(
                    aux_logits.argmax(-1) == (targets[:, :, 1:] - self.aux_start),
                    mask[:, :, 1:]
                )
                acc = torch.cat([acc, aux_acc], dim=2)

            acc_all = acc.float().sum() / mask.float().sum()
            stats["acc_all"] = acc_all.clone().detach()
            
            for idx in range(targets.size(2)):
                weight = mask[:, :, idx].float().sum()
                if weight == 0:
                    stats[f"acc_layer{idx}"] = 0.0
                else:
                    layer_acc = acc[:, :, idx:idx+1].float().sum() 
                    layer_acc = layer_acc / mask[:, :, idx:idx+1].float().sum()
                    stats[f"acc_layer{idx}"] = layer_acc.clone().detach()

        return ce_loss, stats, weight