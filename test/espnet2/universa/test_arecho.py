from test.espnet2.universa.test_metric_tokenizer import token_info

import numpy as np
import pytest
import torch

from espnet2.universa.ar_universa import ARUniversa
from espnet2.universa.ar_universa.data import ARMetricCollateFn, ARMetricProcessor


@pytest.mark.parametrize("pool_size", [1, 10])
def test_arecho_backward_and_constrained_search(pool_size):
    info = token_info()
    model = ARUniversa(
        input_size=8,
        metric2id={"mos": 0, "language": 1},
        metric2type={"mos": "numerical", "language": "categorical"},
        metric_vocab_size=len(info["VOCAB"]) + 4,
        metric_token_info=info,
        embedding_size=8,
        embedding_dim=256,
        use_rope=True,
        use_ref_audio=False,
        use_ref_text=False,
        use_normalize=False,
        audio_encoder_params=dict(
            num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
        ),
        metric_decoder_params=dict(num_blocks=1, attention_heads=2, linear_units=16),
        cross_attention_params=dict(n_head=2, dropout_rate=0.0),
    )
    processor = ARMetricProcessor(
        metric_token_info=info, metrics_list=["mos", "language"], train=True
    )
    samples = [
        (
            "a",
            processor(
                "a",
                dict(
                    audio=np.random.randn(10, 8).astype(np.float32),
                    metrics={"mos": 1.5, "language": "eng"},
                ),
            ),
        )
    ]
    # Test the tokenizer and collator separately from waveform preprocessing.
    samples[0][1]["audio"] = np.random.randn(10, 8).astype(np.float32)
    _, batch = ARMetricCollateFn()(samples)
    loss, stats, _ = model(**batch)
    assert torch.isfinite(loss) and loss > 0
    torch.testing.assert_close(loss, stats["loss"])
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert model.sos == 2 and model.eos == 3  # Published ARECHO convention.
    model.eval()
    model.set_inference(
        pool_size, ["language", "mos"], True, True, use_fixed_order=True
    )
    result = model.inference(batch["audio"], batch["audio_lengths"])
    assert result["language"][0] in ["eng", "jpn"]
    assert result["mos"][0] in info["tokenizer"]["mos"]

    assert result["token_seq"][0][1::2] == [10, 4]
