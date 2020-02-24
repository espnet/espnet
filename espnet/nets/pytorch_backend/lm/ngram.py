import kenlm
import torch

from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.scorer_interface import PartialScorerInterface

class extendkenlm(kenlm.LanguageModel):
    def score_state(self, words, bos = True, eos = True):
        state = kenlm.State()
        out_state = kenlm.State()
        if bos:
            self.BeginSentenceWrite(state)
        else:
            self.NullContextWrite(state)
        total = 0.0
        for word in words:
            total += self.BaseScore(state, word, out_state)
            state = out_state
        if eos:
            total += self.BaseScore(state, self.vocab.EndSentence(), out_state)
        return total, out_state

class Ngram(ScorerInterface):
    def __init__(self, ngram, token_list, space=" "):
        self.space = space
        self.chardict = token_list
        self.charlen = len(self.chardict)
        self.lm = extendkenlm(ngram)
        self.tmpkenlmstate = kenlm.State()

    def init_state(self, x):
        return []

    def select_state(self, state, i):
        return state
    
    def score_partial(self, string, next_tokens):
        basescore, state = self.lm.score_state(string, bos = False, eos= False)
        scores = torch.full(next_tokens.size(), basescore)
        for i,j in enumerate(next_tokens):
            scores[i] += self.lm.BaseScore(state, self.chardict[j], self.tmpkenlmstate)
        return scores 
