import pytest
import torch

from espnet2.asr.frontend.huggingface import HuggingFaceFrontend

pytest.importorskip("transformers", minversion="4.43.0")


@pytest.mark.parametrize(
    "model, fs",
    [
        ("taiqihe/test-w2v-bert-dummy", 16000),
    ],
)
def test_frontend_backward(model, fs):
    frontend = HuggingFaceFrontend(
        model,
        fs=fs,
        download_dir="./hf_cache",
        load_pretrained=False,
    )
    test_length = 640
    wavs = torch.randn(2, test_length, requires_grad=True)
    lengths = torch.LongTensor([test_length, test_length])
    feats, f_lengths = frontend(wavs, lengths)
    feats.sum().backward()
