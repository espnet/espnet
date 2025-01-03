import pytest
import torch

from espnet2.sds.end_to_end.mini_omni_e2e import MiniOmniE2EModel

pytest.importorskip("pydub")
pytest.importorskip("espnet2.sds.end_to_end.mini_omni.inference")
pytest.importorskip("huggingface_hub")


def test_forward():
    if not torch.cuda.is_available():
        return  # Only GPU supported
    dialogue_model = MiniOmniE2EModel()
    dialogue_model.warmup()
    x = torch.randn(2000, requires_grad=False).cpu().numpy()
    dialogue_model.forward(x, orig_sr=16000)
