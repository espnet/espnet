import numpy as np
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.scorer_interface import PartialScorerInterface


class CTCPrefixScorer(PartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore"""

    def __init__(self, ctc, eos):
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x):
        logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i):
        sc, st = state
        return sc[i], st[i]

    def score_partial(self, y, ids, state, x):
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(presub_score - prev_score, device=y.device)
        return tscore, (presub_score, new_st)
