import sys

import pytest
import torch
from packaging.version import parse as V

try:
    import s3prl  # noqa
    from s3prl.upstream.wav2vec2.wav2vec2_model import (  # noqa
        TransformerSentenceEncoderLayer,
    )

    is_s3prl_available = True
except ImportError:
    is_s3prl_available = False
from espnet2.layers.houlsby_adapter_layer import (
    Houlsby_Adapter,
    HoulsbyTransformerSentenceEncoderLayer,
)

pytest.importorskip("transformers")
is_python_3_8_plus = sys.version_info >= (3, 8)
is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")


@pytest.mark.skipif(
    is_s3prl_available and not is_torch_1_8_plus or not is_python_3_8_plus,
    reason="Not supported",
)
def test_transformers_availability_false():
    if not is_s3prl_available:
        assert (
            HoulsbyTransformerSentenceEncoderLayer is None
        ), HoulsbyTransformerSentenceEncoderLayer


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
def test_Houlsby_Adapter_init():

    adapter = Houlsby_Adapter(
        input_size=64,
        bottleneck=32,
    )
    assert adapter.bottleneck == 32
    assert adapter.houlsby_adapter[0].in_features == 64
    assert adapter.houlsby_adapter[2].out_features == 64


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
def test_Houlsby_Adapter_forward():

    adapter = Houlsby_Adapter(
        input_size=64,
        bottleneck=32,
    )
    x = torch.rand(1, 2, 64)
    output = adapter(x)
    assert output.shape == (1, 2, 64)


@pytest.mark.skipif(
    is_s3prl_available and not is_torch_1_8_plus or not is_python_3_8_plus,
    reason="Not supported",
)
def test_HoulsbyTransformerSentenceEncoderLayer_init():
    embedding_dim = 768
    ffn_embedding_dim = 3072
    num_attention_heads = 8
    dropout = 0.1
    attention_dropout = 0.1
    activation_dropout = 0.1
    activation_fn = "relu"
    layer_norm_first = False
    bottleneck = 32
    adapter_added_layer = HoulsbyTransformerSentenceEncoderLayer(
        embedding_dim=embedding_dim,
        ffn_embedding_dim=ffn_embedding_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        activation_fn=activation_fn,
        layer_norm_first=layer_norm_first,
        bottleneck=bottleneck,
    )

    assert adapter_added_layer.bottleneck == 32
    assert embedding_dim == adapter_added_layer.embedding_dim
    assert ffn_embedding_dim == adapter_added_layer.fc1.out_features
    assert num_attention_heads == adapter_added_layer.self_attn.num_heads
    assert dropout == adapter_added_layer.dropout1.p
    assert attention_dropout == adapter_added_layer.self_attn.dropout_module.p
    assert activation_dropout == adapter_added_layer.dropout2.p
    assert activation_fn == adapter_added_layer.activation_fn.__name__
    assert layer_norm_first == adapter_added_layer.layer_norm_first


@pytest.mark.skipif(
    is_s3prl_available and (not is_torch_1_8_plus) or not is_python_3_8_plus,
    reason="Not supported",
)
def test_HoulsbyTransformerSentenceEncoderLayer_forward():
    embedding_dim = 768
    ffn_embedding_dim = 3072
    num_attention_heads = 8
    dropout = 0.1
    attention_dropout = 0.1
    activation_dropout = 0.1
    activation_fn = "relu"
    layer_norm_first = False
    bottleneck = 32
    adapter_added_layer = HoulsbyTransformerSentenceEncoderLayer(
        embedding_dim=embedding_dim,
        ffn_embedding_dim=ffn_embedding_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        activation_fn=activation_fn,
        layer_norm_first=layer_norm_first,
        bottleneck=bottleneck,
    )
    x = torch.rand(1, 2, embedding_dim)
    output, _ = adapter_added_layer(x)
    assert output.shape == (1, 2, embedding_dim)


if __name__ == "__main__":
    test_transformers_availability_false()
    test_Houlsby_Adapter_init()
    test_Houlsby_Adapter_forward()
    test_HoulsbyTransformerSentenceEncoderLayer_init()
    test_Houlsby_Adapter_init()
    print("Houlsby_Adapter and HoulsbyTransformerSentenceEncoderLayer are tested")
