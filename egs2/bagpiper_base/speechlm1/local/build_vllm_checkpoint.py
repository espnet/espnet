import torch
import sys

input_ckpt = sys.argv[1]
output_ckpt = input_ckpt.replace(".pt", "")

ckpt = torch.load(input_ckpt, map_location="cpu")['module']

# (1) remove irrelevant items
remove_keys = list()
for key, value in ckpt.items():
    for prefix in ["multimodal_io", "adaptor", "stream_emb"]:
        if prefix in key:
            remove_keys.append(key)

for key in remove_keys:
    del ckpt[key]

# (2) override text embeddings
input_name = "model.embed_tokens.weight"
output_name = "lm_head.weight"
input_emb = ckpt[input_name]
output_emb = ckpt[output_name]

ckpt[input_name] = input_emb[256: 256 + 151936]
ckpt[output_name] = output_emb[256: 256 + 151936]

# (3) override special tokesn
# <bos><user><text> ... <eos><assistant><text> ... <eos>
for src, tgt in [
    (1, 151644), # <|bos|> -> '<|im_start|>'
    (2, 151643), # <|eos|> -> '<|endoftext|>'
    (7, 151646), # <|text|> -> '<|object_ref_start|>'
    (5, 151645), # <|user|> -> '<|im_end|>'
    (6, 151647), # <|assistant|> -> '<|object_ref_end|>'
]:
    ckpt[input_name][tgt] = input_emb[src]
    ckpt[output_name][tgt] = output_emb[src]

# (4) Save checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")

model.load_state_dict(ckpt)
model.generation_config.max_new_tokens = 16384
model.save_pretrained(output_ckpt)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
tokenizer.save_pretrained(output_ckpt)