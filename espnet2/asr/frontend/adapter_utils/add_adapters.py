import numpy as np

from espnet2.asr.frontend.adapter_utils.adapters.adapter_transformer import (
    AdapterTransformerSentenceEncoderLayer,
)


def add_adapters(s3prl_upstream_model, adapter_down_dim, adapt_layers=None):
    if adapt_layers == []:
        print(">> adapt_layers is an empy list. No adapters will be inserted.")
        return s3prl_upstream_model
    orig_param_num = count_params(s3prl_upstream_model)

    # freeze all layers
    for param in s3prl_upstream_model.parameters():
        param.requires_grad = False

    for layer_idx, layer in enumerate(s3prl_upstream_model.model.encoder.layers):
        if adapt_layers is not None and layer_idx not in adapt_layers:
            continue

        # extract arguments from original layer
        embedding_dim = layer.embedding_dim
        ffn_embedding_dim = layer.fc1.out_features
        num_attention_heads = layer.self_attn.num_heads
        dropout = layer.dropout1.p
        attention_dropout = layer.self_attn.dropout_module.p
        activation_dropout = layer.dropout2.p
        activation_fn = layer.activation_fn.__name__
        layer_norm_first = layer.layer_norm_first

        # initialize adapter-added transformer layer
        adapter_added_layer = AdapterTransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layer_norm_first=layer_norm_first,
            adapter_down_dim=adapter_down_dim,
        )

        # freeze non-adapter layers
        for name, param in adapter_added_layer.named_parameters():
            if "adapter1" in name or "adapter2" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # copy weights
        orig_state_dict = layer.state_dict()
        adapter_added_layer.load_state_dict(orig_state_dict, strict=False)

        # overwrite original layer with adapter-added layer
        s3prl_upstream_model.model.encoder.layers[layer_idx] = adapter_added_layer

    # print resulting model stats
    new_param_num = count_params(s3prl_upstream_model)
    new_trainable_param_num = count_params(s3prl_upstream_model, only_trainable=True)

    print(f"  * original model weights: {orig_param_num:,}")
    print(f"  * new model weights - all: {new_param_num:,}")
    print(
        f"  * new model weights - trainable: {new_trainable_param_num:,} ({100. * new_trainable_param_num / orig_param_num : .2f}% of original model)"
    )

    return s3prl_upstream_model


def count_params(model, only_trainable=False):
    n_params = 0
    for param in model.parameters():
        n = np.prod(param.shape)
        if only_trainable:
            if param.requires_grad:
                n_params += n
        else:
            n_params += n
    return n_params
