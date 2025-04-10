import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.beats_encoder import BeatsConfig, BeatsEncoder  # noqa

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


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
def test_forward_pass(
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
    assert output_rep.size(2) == 768, "Output dim should be 768"

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
def test_backward_pass(
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
    x = torch.randn((2, 32_000), requires_grad=True)  # B,T
    x_lens = torch.LongTensor([16_000, 24_000])
    output_rep, output_len, _ = beats_model(x, x_lens)

    output_rep.sum().backward()


def test_small_inputs():
    if not is_torch_1_12_1_plus:
        return
    beats_config = {"encoder_layers": 2}  # Smaller model
    beats_model = BeatsEncoder(
        input_size=1,
        beats_config=beats_config,
    )
    min_input_len_ = beats_model.min_input_length_at_16khz
    x = torch.randn(2, min_input_len_ // 2, requires_grad=True)
    x_lens = torch.LongTensor([min_input_len_ // 2, min_input_len_ // 4])
    audio_rep, l, _ = beats_model(x, x_lens)  # forward pass should not raise any error
    audio_rep.sum().backward()  # backward pass should not raise any error
