import kenlm
import torch

from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.scorer_interface import PartialScorerInterface

from abc import ABC


class Ngrambase(ABC):
    """Ngram base implemented throught ScorerInterface
    """

    def __init__(self, ngram_model, token_list):
        """
        Args:
            ngram_model: ngram model path
            token_list: token list from dict or model.json
        """
        self.chardict = [x if x != "<eos>" else "</s>" for x in token_list]
        self.charlen = len(self.chardict)
        self.lm = kenlm.LanguageModel(ngram_model)
        self.tmpkenlmstate = kenlm.State()

    def init_state(self, x):
        """
        """
        state = kenlm.State()
        # since there is no <s> only </s>
        self.lm.NullContextWrite(state)
        return [0.0, state]

    def select_state(self, state, i):
        """
        """
        return state

    def score_partial_(self, y, next_token, state, x):
        """
        Args:
            y: previous char
            next_token: next token need to be score
            state: previous state
            x: encoded feature
        """
        out_state = kenlm.State()
        state[0] += self.lm.BaseScore(state[1], self.chardict[y[-1]], out_state)
        scores = torch.full(next_token.size(), state[0])
        for i, j in enumerate(next_token):
            scores[i] += self.lm.BaseScore(
                out_state, self.chardict[j], self.tmpkenlmstate
            )
        return scores, [state[0], out_state]


class NgramFullScorer(Ngrambase, ScorerInterface):
    """
    Args:
        Fullscorer for ngram
    """

    def score(self, y, state, x):
        return self.score_partial_(y, torch.tensor(range(len(self.chardict))), state, x)


class NgramPartScorer(Ngrambase, PartialScorerInterface):
    """
    Args:
        Partialscorer for ngram
    """

    def score_partial(self, y, next_token, state, x):
        return self.score_partial_(y, next_token, state, x)
