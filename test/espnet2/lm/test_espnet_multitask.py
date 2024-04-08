import pytest
import torch

from espnet2.lm.espnet_model_multitask import ESPnetMultitaskLanguageModel
from espnet2.lm.transformer_lm import TransformerLM


@pytest.mark.parametrize("arch", [TransformerLM])
def test_espnet_model_multitask(arch):
    vocab_size = 6
    token_list = ["<blank>", "<unk>", "a", "i", "<sos/eos>", "<generatetext>"]
    sos_syms = ["<generatetext>"]

    lm = arch(vocab_size, layer=2)

    model = ESPnetMultitaskLanguageModel(lm, vocab_size, token_list, sos_syms=sos_syms)

    inputs = dict(
        text=torch.randint(0, 3, [2, 10], dtype=torch.long),
        text_lengths=torch.tensor([10, 8], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
