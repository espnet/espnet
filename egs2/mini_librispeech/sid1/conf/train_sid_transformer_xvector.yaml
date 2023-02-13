batch_type: folded
batch_size: 64
accum_grad: 1
max_epoch: 30
patience: none
# The initialization method for model parameters
init: xavier_uniform
best_model_criterion:
-   - valid
    - acc
    - max
keep_nbest_models: 3

encoder: transformer
encoder_conf:
    output_size: 128
    attention_heads: 4
    linear_units: 256
    num_blocks: 2
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: linear
    normalize_before: true

decoder: transformer
decoder_conf:
    attention_heads: 4
    linear_units: 256
    num_blocks: 1
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.0
    src_attention_dropout_rate: 0.0

model_conf:
    ctc_weight: 0.0
    lsm_weight: 0.0
    length_normalized_loss: false

optim: adam
optim_conf:
    lr: 0.001
