#!/usr/bin/env python3

import sys

import torch
from transformers import AutoModelForCausalLM

model_id = sys.argv[1]
output_path = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(model_id)
state_dict = {"ctc.ctc_lo.weight": model.model.embed_tokens.weight.detach()}
torch.save(state_dict, output_path)
