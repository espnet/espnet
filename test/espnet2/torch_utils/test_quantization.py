import torch

from espnet2.torch_utils.quantization import ensure_quantized_engine, quantize_dynamic


def test_ensure_quantized_engine_picks_an_available_engine():
    engine = ensure_quantized_engine()
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if supported:
        assert engine != "none"
        assert engine in supported
    else:
        assert engine == "none"


def test_quantize_dynamic_runs():
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not supported:
        return  # nothing can be quantized on this build
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).eval()
    quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        out = quantized(torch.randn(2, 8))
    assert out.shape == (2, 8)
