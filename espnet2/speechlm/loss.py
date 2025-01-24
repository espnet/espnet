import torch
import logging
from espnet2.speechlm.net_utils import length_mask
import torch.distributed

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers import LigerCrossEntropyLoss
except:
    LigerFusedLinearCrossEntropyLoss = None
    LigerCrossEntropyLoss = None

class SpeechLMCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        pad,
        vocab_size,
        token_bias,
        modality_weights,
        lm_head: torch.nn.Linear = None,
        aux_lm_head: torch.nn.Linear = None,
        use_liger_kernel: bool = False
    ):
        """
        Compute the CrossEntropy for SpeechLM. The main motivation of this module is to save computing.
        From the second layer and then on, the target codes can only be the codec codes or paddings, 
        which helps us to shrink the vocabulary during the loss computing. 
        """
        super().__init__()

        self.pad = pad
        self.use_aux_ce_loss = "codec" in token_bias
        self.use_liger_kernel = use_liger_kernel

        # (1) parse the weights of tokens
        token_bias = token_bias.copy()
        if self.use_aux_ce_loss:
            self.aux_start, self.aux_end = token_bias["codec"]
        else:
            self.aux_start, self.aux_end = 0, 0

        # prepare the weight for first-layer
        weight = torch.ones(vocab_size).float()
        for modality_name, modality_weight in modality_weights.items():

            if modality_name not in token_bias:
                logging.warning(f"The specified modality {modality_name} is not in token_bias. Skip it")
                continue
            
            start, end = token_bias[modality_name]
            del token_bias[modality_name]
            weight[start: end] = modality_weight

        for modality in token_bias.keys():
            logging.warning(f"weight for modality {modality} not specified. Set to 1.0")
        self.weight = weight
        
        # (2) create CE loss module
        assert lm_head is not None
        self.lm_head = lm_head

        if self.use_aux_ce_loss:
            assert aux_lm_head is not None
            self.aux_lm_head = aux_lm_head

        if use_liger_kernel:
            if LigerCrossEntropyLoss is None:
                raise ValueError('LigerFusedLinearCrossEntropyLoss is not installed')
            ce_loss_imp = LigerCrossEntropyLoss
        else:
            ce_loss_imp = torch.nn.CrossEntropyLoss

        self.ce_loss = ce_loss_imp(
            ignore_index=self.pad,
            reduction='none'
        )
        if self.use_aux_ce_loss:
            self.aux_ce_loss = ce_loss_imp(
                ignore_index=self.pad - self.aux_start,
                reduction='none'
            )
    
    def forward(
        self, 
        logits: torch.Tensor,
        targets: torch.Tensor, 
        loss_mask: torch.Tensor,
    ):
        # NOTE(Jinchuan): keep the weight on the correct device in the first forward.
        # We don't want to keep the weights registered as model parameters as they 
        # should always be specified by external configurations.
        device, dtype = logits[0].device, logits[0].dtype
        self.weight = self.weight.to(device).to(dtype)

        # sanity check
        logits, aux_logits = logits
        assert logits.dim() == 4 and logits.size(2) == 1
        B, T, _, _ = logits.size()
        
        if aux_logits is not None:
            assert logits.dim() == 4
            assert logits.size()[:2] == aux_logits.size()[:2]
            assert logits.size(2) + aux_logits.size(2) == targets.size(2)
        
        # element-wise loss
        _targets = targets[:, :, :1].flatten()
        if self.use_liger_kernel:
            logits = self.lm_head(logits)
            ce_loss = self.ce_loss(
                logits.flatten(end_dim=2),
                _targets,
            )
        else:
            ce_loss = self.apply_ce_loss(
                logits.flatten(end_dim=2),
                _targets,
                self.lm_head,
                self.ce_loss,
                chunk_size=16384 * 100,
            )
        ce_loss = ce_loss * self.weight[_targets]
        ce_loss = ce_loss.view(B, T, 1)

        if aux_logits is not None:
            torch.set_printoptions(sci_mode=False, threshold=1e9)
            _targets = targets[:, :, 1:].flatten()
            assert torch.all(torch.logical_or(
                _targets == self.pad,
                torch.logical_and(
                    _targets >= self.aux_start, 
                    _targets < self.aux_end,
                )
            ))
            if self.use_liger_kernel:
                aux_logits = self.aux_lm_head(aux_logits)
                aux_ce_loss = self.aux_ce_loss(
                    aux_logits.flatten(end_dim=2),
                    _targets - self.aux_start
                )
            else:
                aux_ce_loss = self.apply_ce_loss(
                    aux_logits.flatten(end_dim=2),
                    _targets - self.aux_start,
                    self.aux_lm_head,
                    self.aux_ce_loss,
                    chunk_size=32768 * 100 ,
                )
            aux_ce_loss = aux_ce_loss * self.weight[_targets]
            aux_ce_loss = aux_ce_loss.view(B, T, -1)

            ce_loss = torch.cat([ce_loss, aux_ce_loss], dim=2)
        
        ce_loss = ce_loss * loss_mask
        weight = loss_mask[..., 0].eq(1).float().sum()

        ce_loss = ce_loss.sum() / weight
        stats = {"ce_loss": ce_loss.clone().detach(), "weight": weight.clone().detach()}

        # logging, if not training
        if not self.training:
            logits = logits if self.use_liger_kernel else self.lm_head(logits)
            acc = logits.argmax(-1) == targets[:, :, :1]
            if aux_logits is not None:
                aux_logits = aux_logits if self.use_liger_kernel else self.aux_lm_head(aux_logits)
                aux_acc = aux_logits.argmax(-1) == targets[:, :, 1:] - self.aux_start
                acc = torch.cat([acc, aux_acc], dim=2)
            
            acc = torch.where(loss_mask.bool(), acc, False)

            acc_all = acc.float().sum() / loss_mask.float().sum()
            stats["acc_all"] = acc_all.clone().detach()
            
            for idx in range(targets.size(2)):
                layer_weight = loss_mask[:, :, idx].float().sum()
                if layer_weight == 0:
                    stats[f"acc_layer{idx}"] = 0.0
                else:
                    layer_acc = acc[:, :, idx:idx+1].float().sum() 
                    layer_acc = layer_acc / loss_mask[:, :, idx:idx + 1].float().sum()
                    stats[f"acc_layer{idx}"] = layer_acc.clone().detach()
        
        return ce_loss, stats, weight
    
    def apply_ce_loss(self, 
        input, 
        target,
        linear_module,
        loss_module,
        chunk_size=10000
    ):
        # Apply CE loss chunk-by-chunk to avoid memory spike

        assert input.dim() == 2
        assert target.dim() == 1

        start = 0
        ce_loss = []
        while start < input.size(0):
            end = min(input.size(0), start + chunk_size)
            piece_ce_loss = loss_module(
                linear_module(input[start: end]),
                target[start: end],
            )
            ce_loss.append(piece_ce_loss)
            start += chunk_size

        return torch.cat(ce_loss)

