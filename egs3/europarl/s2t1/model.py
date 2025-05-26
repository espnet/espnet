import torch
import torch.nn as nn

from espnet2.bin.s2t_inference import Speech2Text


class OWSMFinetune(nn.Module):
    def __init__(self, model_tag):
        super().__init__()
        owsm_model = Speech2Text.from_pretrained(model_tag)
        self.model = owsm_model.s2t_model
        self.model.train()

    def forward(
        self,
        speech,
        speech_lengths,
        text,
        text_lengths,
        text_ctc,
        text_ctc_lengths,
        text_prev,
        text_prev_lengths,
    ):
        return self.model(
            speech,
            speech_lengths,
            text,
            text_lengths,
            text_ctc,
            text_ctc_lengths,
            text_prev,
            text_prev_lengths,
        )

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}
