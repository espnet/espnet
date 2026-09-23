"""Check copied LM task construction against its ESPnet2 source."""

import pytest
import torch

from espnet3.utils.task_utils import get_espnet_model


@pytest.mark.parametrize(
    "lm,model_config",
    [
        ("seq_rnn", {"unit": 8, "nlayers": 1}),
        (
            "transformer",
            {"embed_unit": 8, "att_unit": 8, "head": 2, "unit": 16, "layer": 1},
        ),
    ],
)
def test_source_models_and_losses_are_preserved(lm, model_config):
    """Keep both LM architectures usable without changing their model behavior."""
    config = {
        "token_list": ["<blank>", "<unk>", "HELLO", "WORLD", "<sos/eos>"],
        "lm": lm,
        "lm_conf": model_config,
    }
    torch.manual_seed(0)
    original = get_espnet_model("espnet2.tasks.lm.LMTask", config).eval()
    torch.manual_seed(0)
    migrated = get_espnet_model(
        "espnet3.systems.esp2_asr.lm_task.LMTask", config
    ).eval()
    inputs = {"text": torch.tensor([[2, 3]]), "text_lengths": torch.tensor([2])}
    torch.testing.assert_close(original(**inputs)[0], migrated(**inputs)[0])
    assert original.state_dict().keys() == migrated.state_dict().keys()
