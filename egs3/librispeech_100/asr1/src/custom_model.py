import torch
import torch.nn as nn

from espnet2.bin.s2t_inference import Speech2Text



class OWSMFinetuneModel(nn.Module):

    def __init__(self):
        super().__init__()
        # About 30 sec
        pretrained_model = Speech2Text.from_pretrained(
            "espnet/owsm_v4_base_102M",
            lang_sym=f"<eng>",
            beam_size=1,
            device='cuda'
        )

        self.s2t_model = pretrained_model.s2t_model

    def forward(
            self, 
            speech, speech_lengths,
            text, text_lengths,
            text_ctc=None, text_ctc_lengths=None,
            text_prev=None, text_prev_lengths=None
        ):
        # OWSM (s2t_model) の forward にそのまま流す
        loss, stats, weight = self.s2t_model(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            text_ctc=text_ctc,
            text_ctc_lengths=text_ctc_lengths,
            text_prev=text_prev,
            text_prev_lengths=text_prev_lengths,
        )

        return loss, stats, weight
