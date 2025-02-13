import pytest
import torch
from packaging.version import parse as V
import time

from espnet2.asr.encoder.beats_encoder import (
    BeatsEncoder,
    BeatsPretrainingPredictor,
    MultiheadAttention,
)

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")
is_torch_2_plus = V(torch.__version__) >= V("2.0.0")


def test_override_beats_config():
    if not is_torch_1_12_1_plus:
        return

    beats_config = {"encoder_layers": 2}
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
    )
    assert (
        len(beats_model.encoder.layers) == 2
    ), f"Number of layers should be 2. It is {len(beats_model.encoder.layers)}"


# Each parameter value creates a variant of the model
@pytest.mark.timeout(30)
@pytest.mark.parametrize("downsampling_rate", [1, 2])
@pytest.mark.parametrize("use_weighted_representation", [False, True])
@pytest.mark.parametrize(
    "add_positional_information, max_positions", [(False, None), (True, 12800)]
)
def test_forward_pass_beats_encoder(
    downsampling_rate,
    use_weighted_representation,
    add_positional_information,
    max_positions,
):
    if not is_torch_1_12_1_plus:
        return

    beats_config = {"encoder_layers": 2}  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        downsampling_rate=downsampling_rate,
        use_weighted_representation=use_weighted_representation,
        add_positional_information=add_positional_information,
        max_positions=max_positions,
    )
    x = torch.randn((2, 32_000))  # B,T
    x_lens = torch.LongTensor([16_000, 32_000])
    output_rep, output_len, _ = beats_model(x, x_lens)

    # Check output representation
    assert (
        output_rep.size(0) == 2
    ), f"Representation batch size should be 2. It is {output_rep.size(0)}"
    correct_length = 47 if downsampling_rate == 2 else 96
    assert (
        output_rep.size(1) == correct_length
    ), f"Representation length should be {correct_length}. It is {output_rep.size(1)}"
    assert output_rep.size(2) == 768, f"Output dim should be 768"

    # Check output length
    assert (
        output_len.size(0) == 2
    ), f"Output length batch size should be 2. It is {output_len.size(0)}"
    correct_length = (24, 47) if downsampling_rate == 2 else (48, 96)
    assert (
        tuple(output_len.tolist()) == correct_length
    ), f"Output length vector should be {correct_length}. It is {output_len}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("downsampling_rate", [1, 2])
@pytest.mark.parametrize("use_weighted_representation", [False, True])
@pytest.mark.parametrize(
    "add_positional_information, max_positions", [(False, None), (True, 12800)]
)
def test_backward_pass_beats_encoder(
    downsampling_rate,
    use_weighted_representation,
    add_positional_information,
    max_positions,
):
    if not is_torch_1_12_1_plus:
        return

    beats_config = {"encoder_layers": 2}  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        downsampling_rate=downsampling_rate,
        use_weighted_representation=use_weighted_representation,
        add_positional_information=add_positional_information,
        max_positions=max_positions,
    )
    x = torch.randn((2, 24_000), requires_grad=True)  # B,T
    x_lens = torch.LongTensor([16_000, 24_000])
    output_rep, output_len, _ = beats_model(x, x_lens)

    output_rep.sum().backward()


# Test for pretraining mode


def test_forward_pass_pretraining_beats_encoder():
    beats_config = {"encoder_layers": 2, "mask_ratio": 0.75}
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    x = torch.randn((2, 32_000))  # B,T
    x_lens = torch.LongTensor([16_000, 32_000])
    output_rep, patch_len, restore_ids, kept_mask = beats_model(x, x_lens)

    # Check output representation
    assert (
        output_rep.size(0) == 2
    ), f"Representation batch size should be 2. It is {output_rep.size(0)}"
    correct_length = 96 // 4  # with 0.75 masking rate
    assert (
        output_rep.size(1) == correct_length
    ), f"Representation length should be {correct_length}. It is {output_rep.size(1)}"
    assert output_rep.size(2) == 768, f"Output dim should be 768"
    assert tuple(patch_len.tolist()) == (48, 96)

    assert kept_mask.shape == (
        2,
        96,
    ), f"Kept mask shape should be (2, 96). It is {kept_mask.shape}"

    assert restore_ids.shape == (
        2,
        96,
    ), f"Restore ids shape should be (2, 96). It is {restore_ids.shape}"
    assert (
        restore_ids.sort(dim=1).values.tolist()
        == torch.arange(96).unsqueeze(0).repeat(2, 1).tolist()
    ), "Restore ids should be a permutation"


def test_backward_pass_pretraining_beats_encoder():
    beats_config = {"encoder_layers": 2, "mask_ratio": 0.75}
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    x = torch.randn((2, 32_000), requires_grad=True)  # B,T,1
    x_lens = torch.LongTensor([16_000, 32_000])
    output_rep, patch_len, restore_ids, kept_mask = beats_model(x, x_lens)
    output_rep.sum().backward()


def test_forward_pass_beats_pretraining_predictor():
    if not is_torch_1_12_1_plus:
        return
    beats_config = {
        "encoder_layers": 2,
        "encoder_embed_dim": 128,
        "decoder_embed_dim": 1024,
        "encoder_attention_heads": 4,
        "codebook_vocab_size": 24,
    }
    beats_predictor = BeatsPretrainingPredictor(
        beats_config=beats_config,
    )
    unmasked_patch_rep = torch.randn((2, 24, 128))  # B,T_small,D
    patch_len = torch.LongTensor([96, 96])
    restore_ids = torch.stack([torch.randperm(96) for _ in range(2)])
    pred = beats_predictor(unmasked_patch_rep, patch_len, restore_ids)
    assert pred.size(0) == 2, f"Batch size should be 2. It is {pred.size(0)}"
    assert pred.size(1) == 96, f"Output length should be 96. It is {pred.size(1)}"
    assert pred.size(2) == 24, f"Output dim should be 24. It is {pred.size(2)}"


def test_backward_pass_beats_pretraining_predictor():
    if not is_torch_1_12_1_plus:
        return
    beats_config = {
        "decoder_layers": 2,
        "encoder_layers": 2,
        "encoder_embed_dim": 128,
        "decoder_embed_dim": 1024,
        "encoder_attention_heads": 4,
        "codebook_vocab_size": 24,
    }
    beats_predictor = BeatsPretrainingPredictor(
        beats_config=beats_config,
    )
    unmasked_patch_rep = torch.randn((2, 24, 128), requires_grad=True)  # B,T_small,D
    patch_len = torch.LongTensor([96, 96])
    restore_ids = torch.stack([torch.randperm(96) for _ in range(2)])
    pred = beats_predictor(unmasked_patch_rep, patch_len, restore_ids)
    pred.sum().backward()


@pytest.mark.parametrize("key_padding_mask", [True, None])
@pytest.mark.parametrize("has_relative_attention_bias", [True, False])
@pytest.mark.parametrize("gru_rel_pos", [True, False])
@pytest.mark.parametrize("variable_length", [True, False])
def test_flash_attn(
    key_padding_mask, has_relative_attention_bias, gru_rel_pos, variable_length
):
    if not is_torch_2_plus or not torch.cuda.is_available():
        return
    attn_module = (
        MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=320 if has_relative_attention_bias else 0,
            max_distance=800 if has_relative_attention_bias else 0,
            rescale_init=False,
            gru_rel_pos=gru_rel_pos,
        )
        .to("cuda")
        .to(torch.float16)
    )

    query = torch.rand(128, 32, 512, dtype=torch.float16, device="cuda")
    key = torch.rand(128, 32, 512, dtype=torch.float16, device="cuda")
    value = torch.rand(128, 32, 512, dtype=torch.float16, device="cuda")
    if key_padding_mask:
        if not variable_length:
            key_padding_mask = torch.zeros(32, 128, dtype=torch.bool, device="cuda")
        else:
            key_padding_mask = torch.zeros(32, 128, dtype=torch.bool, device="cuda")
            key_padding_mask[0, 65:] = 1

    # import time
    # start = time.time()
    v1, _, _ = attn_module(
        query=query,
        key=key,
        value=value,
        need_weights=False,
        key_padding_mask=key_padding_mask,
    )
    # print("Time taken vanilla attn: ", time.time()-start)

    attn_module.use_flash_attn = True
    # start=time.time()
    v2, _, _ = attn_module(
        query=query,
        key=key,
        value=value,
        need_weights=False,
        key_padding_mask=key_padding_mask,
    )
    # print("Time taken Flash attn: ", time.time()-start)

    assert v1.shape == v2.shape
    assert torch.allclose(v1, v2, atol=1e-3)


def test_mask_sequence_no_padding():
    beats_config = {"encoder_layers": 1, "mask_ratio": 0.75}
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    N, L, D = 3, 20, 4
    x = torch.randn(N, L, D)
    padding_mask = torch.zeros(N, L, dtype=torch.bool)  # No padding
    x_unmasked, padding_mask, ids_restore, kept = beats_model.mask_sequence(
        x, padding_mask
    )

    num_keep = int(L * (1 - beats_model.mask_ratio))
    # match shapes
    assert x_unmasked.shape == (N, num_keep, D)
    assert padding_mask.shape == (N, num_keep)
    assert ids_restore.shape == (N, L)
    assert kept.shape == (N, L)

    # ensure correct num of tokens are masked,
    # for full length sequences
    num_unmasked = (kept).sum(dim=1)
    assert torch.all(num_unmasked == num_keep)


@pytest.mark.parametrize("variable_seq_len", [True, False])
def test_mask_sequence_ids_restore(variable_seq_len):
    """Ensure ids_restore correctly maps masked tokens back"""
    beats_config = {"encoder_layers": 1, "mask_ratio": 0.75}
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    N, L, D = 5, 23, 7
    x = torch.randn(N, L, D)
    padding_mask = torch.zeros(N, L, dtype=torch.bool)
    if variable_seq_len:
        padding_mask[0, 15:] = True
        padding_mask[1, 10:] = True
        padding_mask[2, 5:] = True
        padding_mask[3, 15:] = True
    x_unmasked, _, ids_restore, kept = beats_model.mask_sequence(x, padding_mask)

    # Restore sequence
    restore_extra_len = L - x_unmasked.shape[1]
    x_restored = torch.cat(
        [x_unmasked, torch.zeros_like(x[:, :restore_extra_len, :])], dim=1
    )
    x_restored = torch.gather(
        x_restored, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
    )
    # Ensure unmasked values are restored correctly
    assert torch.allclose(x_restored[kept], x[kept])


def test_mask_sequence_var_length():
    beats_config = {"encoder_layers": 1, "mask_ratio": 0.75}
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    N, L, D = 3, 20, 4
    L1, L2, L3 = 15, 20, 10
    x = torch.randn(N, L, D)
    padding_mask = torch.zeros(N, L, dtype=torch.bool)
    padding_mask[0, L1:] = True
    padding_mask[1, L2:] = True
    padding_mask[2, L3:] = True
    x_unmasked, padding_mask, ids_restore, kept = beats_model.mask_sequence(
        x, padding_mask
    )

    num_keep_max = int(round(max(L1, L2, L3) * (1 - beats_model.mask_ratio)))
    # match shapes
    assert x_unmasked.shape == (N, num_keep_max, D)
    assert padding_mask.shape == (N, num_keep_max)
    assert ids_restore.shape == (N, L)
    assert kept.shape == (N, L)

    # Ensure correct number of tokens are masked
    num_keep_expected = (
        (torch.tensor([L1, L2, L3], dtype=torch.long) * (1 - beats_model.mask_ratio))
        .round()
        .to(torch.long)
    )
    assert torch.all(num_keep_expected == kept.sum(dim=1))
