import numpy
import torch

from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


def test_decoder_cache():
    adim = 4
    odim = 5
    decoder = Decoder(
        odim=odim,
        attention_dim=adim,
        linear_units=3,
        num_blocks=2,
        dropout_rate=0.0)
    dlayer = decoder.decoders[0]
    memory = torch.randn(2, 5, adim)

    x = torch.randn(2, 5, adim)
    mask = subsequent_mask(x.shape[1]).unsqueeze(0)
    prev_mask = mask[:, :-1, :-1]
    with torch.no_grad():
        torch.manual_seed(0)
        numpy.random.seed(0)
        # layer-level test
        y = dlayer(x, mask, memory, None)[0]
        cache = dlayer(x[:, :-1], prev_mask, memory, None)[0]
        y_fast = dlayer(x, mask, memory, None, cache=cache)[0]
        numpy.testing.assert_allclose(y.numpy(), y_fast.numpy(), rtol=1e-5)

        # decoder-level test
        x = torch.randint(0, odim, x.shape[:2])
        y = decoder.recognize(x, mask, memory)
        y_, cache = decoder.recognize(x[:, :-1], prev_mask, memory, cache=decoder.init_cache())
        y_fast = decoder.recognize(x, mask, memory, cache=None)
        numpy.testing.assert_allclose(y.numpy(), y_fast.numpy(), rtol=0.1)
