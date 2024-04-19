from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list, th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)


class ESPnetMultitaskLanguageModel(AbsESPnetModel):
    @typechecked
    def __init__(
        self,
        lm: AbsLM,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        ignore_id: int = 0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        sos_syms: List[str] = ["<generatetext>", "<generatespeech>"],
        eos_sym: str = "<sos/eos>",
    ):
        super().__init__()
        self.lm = lm
        self.sos_ids = [token_list.index(t) for t in sos_syms]
        self.eos_id = token_list.index(eos_sym)

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

        self.token_list = token_list.copy()

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood (nll)

        NOTE(yifan): We only use nll to calculate perplexity,
            so there is no condition in each sentence.

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        assert max_length is None

        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # NOTE(yifan): The first token is space when using bpe
        text = text[:, 1:]
        text_lengths = text_lengths - 1

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x, x_lengths = text, text_lengths  # text already has <sos>
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.eos_id
        t = t[:, 1:]  # remove <sos>

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            raise NotImplementedError
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths

    def batchify_nll(
        self, text: torch.Tensor, text_lengths: torch.Tensor, batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll) from transformer language model

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase

        """
        total_num = text.size(0)
        if total_num <= batch_size:
            nll, x_lengths = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []
            max_length = text_lengths.max()

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths = self.nll(
                    batch_text, batch_text_lengths, max_length=max_length
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nlls)
            x_lengths = torch.cat(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def _calc_att_loss(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        # NOTE(yifan): The first token is space when using bpe
        text = text[:, 1:]
        text_lengths = text_lengths - 1

        # 1. Prepare input and target
        input = pad_list(
            [t[:t_len] for t, t_len in zip(text, text_lengths)], self.eos_id
        )

        target = []
        for cur_text, cur_text_len in zip(text, text_lengths):
            cur_text = cur_text[:cur_text_len]
            # mask out the condition text
            for sos in self.sos_ids:
                if sos in cur_text:
                    cur_text[: (cur_text == sos).nonzero()[0][0] + 1] = self.ignore_id
                    break
            cur_text = cur_text[1:]  # left shift
            cur_text = F.pad(cur_text, (0, 1), value=self.eos_id)  # add eos
            target.append(cur_text)
        target = pad_list(target, self.ignore_id)

        # 2. Compute attention loss
        pred, _ = self.lm(input, None)

        loss = self.criterion_att(pred, target)
        acc = th_accuracy(
            pred.view(-1, pred.shape[-1]),
            target,
            ignore_label=self.ignore_id,
        )

        return loss, acc

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = text.shape[0]
        loss, acc = self._calc_att_loss(text, text_lengths)
        stats = dict(
            loss=loss.detach(),
            acc=acc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return {}
