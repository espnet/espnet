import copy

import torch
from espnet.nets.scorer_interface import ScorerInterface

class Qwen2HFScorer(ScorerInterface):
    def __init__(self, model, input_ids, attention_mask, input_features, feature_attention_mask):
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.input_features = input_features
        self.feature_attention_mask = feature_attention_mask
        # self.past_kv = None
        # self.init = True

    def init_state(self, xs):
        return {"past_kv": None, "step": 0}

    def score(self, ys, states, xs):
        with torch.no_grad():
            if states["past_kv"] is None:
                # first step: do full prefill
                output = self.model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    input_features=self.input_features,
                    feature_attention_mask=self.feature_attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                logits = output.logits[:, -1]
                new_past = output.past_key_values
                new_step = self.input_ids.size(1)
            else:
                # subsequent step: feed only the last token
                past_len = states["step"]
                new_mask = torch.ones((self.attention_mask.size(0), 1),
                                      dtype=self.attention_mask.dtype,
                                      device=self.attention_mask.device)
                output = self.model(
                    input_ids=ys[-1:].unsqueeze(0),
                    attention_mask=new_mask,
                    input_features=self.input_features,
                    feature_attention_mask=self.feature_attention_mask,
                    past_key_values=states["past_kv"],
                    use_cache=True,
                    return_dict=True,
                )
                logits = output.logits[:, -1]
                new_past = output.past_key_values
                new_step = past_len + 1

        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
        new_states = {"past_kv": copy.deepcopy(new_past), "step": new_step}
        return log_probs, new_states

    def select_state(self, states, idx):
        # states is a list of state-dicts for all beams
        # idx is the list of beam indices you kept
        return [states[i] for i in idx]
    
    def final_score(self, states):
        return 0.0