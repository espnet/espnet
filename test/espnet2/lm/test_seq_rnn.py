import torch

from espnet2.lm.seq_rnn import SequentialRNNLM


def test_SequentialRNNLM_backward():
    model = SequentialRNNLM(10)
    input = torch.randint(0, 9, [2, 10])

    out, h = model(input, None)
    out, h = model(input, h)
    out.sum().backward()
