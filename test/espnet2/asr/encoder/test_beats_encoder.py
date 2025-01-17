import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.beats_encoder import (
    BeatsConfig,
    BeatsEncoder,
    BeatsPretrainingPredictor,
)

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


def test_override_beats_config():
    if not is_torch_1_12_1_plus:
        return

    beats_config = BeatsConfig(cfg={"encoder_layers": 2})
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

    beats_config = BeatsConfig(cfg={"encoder_layers": 2})  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        downsampling_rate=downsampling_rate,
        use_weighted_representation=use_weighted_representation,
        add_positional_information=add_positional_information,
        max_positions=max_positions,
    )
    x = torch.randn((2, 32_000, 1))  # B,T,1
    x_lens = torch.LongTensor([16_000, 24_000])
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
    correct_length = (25, 38) if downsampling_rate == 2 else (50, 75)
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

    beats_config = BeatsConfig(cfg={"encoder_layers": 2})  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        downsampling_rate=downsampling_rate,
        use_weighted_representation=use_weighted_representation,
        add_positional_information=add_positional_information,
        max_positions=max_positions,
    )
    x = torch.randn((2, 32_000, 1), requires_grad=True)  # B,T,1
    x_lens = torch.LongTensor([16_000, 24_000])
    output_rep, output_len, _ = beats_model(x, x_lens)

    output_rep.sum().backward()


# Test for pretraining mode


def test_forward_pass_pretraining_beats_encoder():
    beats_config = BeatsConfig(
        cfg={"encoder_layers": 2, "mask_ratio": 0.75}
    )  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    x = torch.randn((2, 32_000, 1))  # B,T,1
    x_lens = torch.LongTensor([16_000, 24_000])
    output_rep, restore_ids, kept_mask = beats_model(x, x_lens)

    # Check output representation
    assert (
        output_rep.size(0) == 2
    ), f"Representation batch size should be 2. It is {output_rep.size(0)}"
    correct_length = 96 // 4  # with 0.75 masking rate
    assert (
        output_rep.size(1) == correct_length
    ), f"Representation length should be {correct_length}. It is {output_rep.size(1)}"
    assert output_rep.size(2) == 768, f"Output dim should be 768"

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
    beats_config = BeatsConfig(
        cfg={"encoder_layers": 2, "mask_ratio": 0.75}
    )  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
        is_pretraining=True,
    )
    x = torch.randn((2, 32_000, 1), requires_grad=True)  # B,T,1
    x_lens = torch.LongTensor([16_000, 24_000])
    output_rep, restore_ids, kept_mask = beats_model(x, x_lens)
    output_rep.sum().backward()


def test_forward_pass_beats_pretraining_predictor():
    if not is_torch_1_12_1_plus:
        return
    beats_config = BeatsConfig(
        cfg={
            "encoder_layers": 2,
            "encoder_embed_dim": 128,
            "decoder_embed_dim": 1024,
            "encoder_attention_heads": 4,
            "codebook_vocab_size": 24,
        }
    )  # Smaller model
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
    beats_config = BeatsConfig(
        cfg={
            "encoder_layers": 2,
            "encoder_embed_dim": 128,
            "decoder_embed_dim": 1024,
            "encoder_attention_heads": 4,
            "codebook_vocab_size": 24,
        }
    )  # Smaller model
    beats_predictor = BeatsPretrainingPredictor(
        beats_config=beats_config,
    )
    unmasked_patch_rep = torch.randn((2, 24, 128), requires_grad=True)  # B,T_small,D
    patch_len = torch.LongTensor([96, 96])
    restore_ids = torch.stack([torch.randperm(96) for _ in range(2)])
    pred = beats_predictor(unmasked_patch_rep, patch_len, restore_ids)
    pred.sum().backward()
