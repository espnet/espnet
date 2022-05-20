import numpy
import pytest
import torch

from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

RTOL = 1e-4


@pytest.mark.parametrize("normalize_before", [True, False])
def test_decoder_cache(normalize_before):
    adim = 4
    odim = 5
    decoder = Decoder(
        odim=odim,
        attention_dim=adim,
        linear_units=3,
        num_blocks=2,
        normalize_before=normalize_before,
        dropout_rate=0.0,
    )
    dlayer = decoder.decoders[0]
    memory = torch.randn(2, 5, adim)

    x = torch.randn(2, 5, adim) * 100
    mask = subsequent_mask(x.shape[1]).unsqueeze(0)
    prev_mask = mask[:, :-1, :-1]
    decoder.eval()
    with torch.no_grad():
        # layer-level test
        y = dlayer(x, mask, memory, None)[0]
        cache = dlayer(x[:, :-1], prev_mask, memory, None)[0]
        y_fast = dlayer(x, mask, memory, None, cache=cache)[0]
        numpy.testing.assert_allclose(y.numpy(), y_fast.numpy(), rtol=RTOL)

        # decoder-level test
        x = torch.randint(0, odim, x.shape[:2])
        y, _ = decoder.forward_one_step(x, mask, memory)
        y_, cache = decoder.forward_one_step(
            x[:, :-1], prev_mask, memory, cache=decoder.init_state(None)
        )
        y_fast, _ = decoder.forward_one_step(x, mask, memory, cache=cache)
        numpy.testing.assert_allclose(y.numpy(), y_fast.numpy(), rtol=RTOL)


@pytest.mark.parametrize("normalize_before", [True, False])
def test_encoder_cache(normalize_before):
    adim = 4
    idim = 5
    encoder = Encoder(
        idim=idim,
        attention_dim=adim,
        linear_units=3,
        num_blocks=2,
        normalize_before=normalize_before,
        dropout_rate=0.0,
        input_layer="embed",
    )
    elayer = encoder.encoders[0]
    x = torch.randn(2, 5, adim)
    mask = subsequent_mask(x.shape[1]).unsqueeze(0)
    prev_mask = mask[:, :-1, :-1]
    encoder.eval()
    with torch.no_grad():
        # layer-level test
        y = elayer(x, mask, None)[0]
        cache = elayer(x[:, :-1], prev_mask, None)[0]
        y_fast = elayer(x, mask, cache=cache)[0]
        numpy.testing.assert_allclose(y.numpy(), y_fast.numpy(), rtol=RTOL)

        # encoder-level test
        x = torch.randint(0, idim, x.shape[:2])
        y = encoder.forward_one_step(x, mask)[0]
        y_, _, cache = encoder.forward_one_step(x[:, :-1], prev_mask)
        y_fast, _, _ = encoder.forward_one_step(x, mask, cache=cache)
        numpy.testing.assert_allclose(y.numpy(), y_fast.numpy(), rtol=RTOL)


if __name__ == "__main__":
    # benchmark with synth dataset
    from time import time

    import matplotlib.pyplot as plt

    adim = 4
    odim = 5
    model = "decoder"
    if model == "decoder":
        decoder = Decoder(
            odim=odim,
            attention_dim=adim,
            linear_units=3,
            num_blocks=2,
            dropout_rate=0.0,
        )
        decoder.eval()
    else:
        encoder = Encoder(
            idim=odim,
            attention_dim=adim,
            linear_units=3,
            num_blocks=2,
            dropout_rate=0.0,
            input_layer="embed",
        )
        encoder.eval()

    xlen = 100
    xs = torch.randint(0, odim, (1, xlen))
    memory = torch.randn(2, 500, adim)
    mask = subsequent_mask(xlen).unsqueeze(0)

    result = {"cached": [], "baseline": []}
    n_avg = 10
    for key, value in result.items():
        cache = None
        print(key)
        for i in range(xlen):
            x = xs[:, : i + 1]
            m = mask[:, : i + 1, : i + 1]
            start = time()
            for _ in range(n_avg):
                with torch.no_grad():
                    if key == "baseline":
                        cache = None
                    if model == "decoder":
                        y, new_cache = decoder.forward_one_step(
                            x, m, memory, cache=cache
                        )
                    else:
                        y, _, new_cache = encoder.forward_one_step(x, m, cache=cache)
            if key == "cached":
                cache = new_cache
            dur = (time() - start) / n_avg
            value.append(dur)
        plt.plot(range(xlen), value, label=key)
    plt.xlabel("hypothesis length")
    plt.ylabel("average time [sec]")
    plt.grid()
    plt.legend()
    plt.savefig(f"benchmark_{model}.png")
