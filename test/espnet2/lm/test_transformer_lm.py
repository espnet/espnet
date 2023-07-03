import pytest
import torch

from espnet2.lm.transformer_lm import TransformerLM
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch


@pytest.mark.parametrize("pos_enc", ["sinusoidal", None])
def test_TransformerLM_backward(pos_enc):
    model = TransformerLM(10, pos_enc=pos_enc, unit=10)
    input = torch.randint(0, 9, [2, 5])

    out, h = model(input, None)
    out, h = model(input, h)
    out.sum().backward()


@pytest.mark.parametrize("pos_enc", ["sinusoidal", None])
def test_TransformerLM_score(pos_enc):
    model = TransformerLM(10, pos_enc=pos_enc, unit=10)
    input = torch.randint(0, 9, (12,))
    state = model.init_state(None)
    model.score(input, state, None)


def test_TransformerLM_invalid_type():
    with pytest.raises(ValueError):
        TransformerLM(10, pos_enc="fooo")


@pytest.mark.parametrize("pos_enc", ["sinusoidal", None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_TransformerLM_beam_search(pos_enc, dtype):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)
    model = TransformerLM(vocab_size, pos_enc=pos_enc, unit=10)

    beam = BeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": model},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, 20).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=0.0,
            minlenratio=0.0,
        )


@pytest.mark.parametrize("pos_enc", ["sinusoidal", None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_TransformerLM_batch_beam_search(pos_enc, dtype):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)

    model = TransformerLM(vocab_size, pos_enc=pos_enc, unit=10)
    beam = BatchBeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": model},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, 20).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=0.0,
            minlenratio=0.0,
        )
