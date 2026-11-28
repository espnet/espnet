import pytest
import torch

from espnet3.systems.snoar.src.model import SNOARModel
from espnet3.systems.snoar.src.net import load_net, save_net

from .conftest import tiny_net

B, T = 2, 2000


def _batch():
    torch.manual_seed(0)
    return {
        k: torch.randn(B, T)
        for k in ("speech_mix", "noise2", "speech_ref", "noise_ref")
    }


@pytest.mark.parametrize("mode,slots", [("supervised", 1), ("dnf", 2), ("mixit", 3)])
def test_seed_modes_train(mode, slots):
    m = SNOARModel(mode, net=tiny_net(slots))
    loss, stats, weight = m(**_batch())
    loss.backward()
    assert torch.isfinite(loss) and int(weight) == B
    assert any(p.grad is not None for p in m.net.parameters())


def test_snoar_from_exported_seed(tmp_path):
    seed = SNOARModel("dnf", net=tiny_net(2))
    save_net(seed.inference_net(), tmp_path / "seed", {"readout": seed.readout})
    m = SNOARModel(
        "snoar",
        init=str(tmp_path / "seed"),
        steps_per_generation=1,
        n_generations=3,
        bank_size=8,
    )
    assert m.readout == "dnf"  # read from snoar.yaml
    before = [p.clone() for p in m.teacher.parameters()]
    for _ in range(2):
        loss, stats, _ = m(speech_mix=_batch()["speech_mix"])
        loss.backward()
        m.update_teacher()
    assert int(m.num_updates) == 2 and m.generation == 3
    assert stats["gamma"] == pytest.approx(0.125)  # generation 2 of 3, cosine
    assert all(not p.requires_grad for p in m.teacher.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, m.teacher.parameters()))
    m.eval()
    _, stats, _ = m(
        speech_mix=_batch()["speech_mix"], speech_ref=_batch()["speech_ref"]
    )
    assert "si_sdr" in stats and not m.teacher.training


def test_state_dict_round_trip_keeps_banks(tmp_path):
    seed = SNOARModel("supervised", net=tiny_net(1))
    save_net(seed.inference_net(), tmp_path / "seed", {"readout": "raw"})
    m = SNOARModel("snoar", init=str(tmp_path / "seed"), bank_size=8)
    m(speech_mix=torch.randn(B, T))
    m2 = SNOARModel("snoar", init=str(tmp_path / "seed"), bank_size=8)
    m2.load_state_dict(m.state_dict())
    assert int(m2.bank_count) == B and torch.equal(m2.snr_bank, m.snr_bank)
    assert load_net(str(tmp_path / "seed")).num_outputs == 1


def test_invalid_configurations_raise(tmp_path):
    with pytest.raises(ValueError, match="needs a seed"):
        SNOARModel("snoar", net=tiny_net(2))
    with pytest.raises(ValueError, match="needs 2 output slot"):
        SNOARModel("dnf", net=tiny_net(1))
    with pytest.raises(ValueError, match="not in"):
        SNOARModel("bogus", net=tiny_net(1))
    with pytest.raises(ValueError, match="needs `noise2`"):
        SNOARModel("dnf", net=tiny_net(2))(speech_mix=torch.randn(B, T))
    with pytest.raises(FileNotFoundError, match="not an exported model"):
        SNOARModel("snoar", init=str(tmp_path))


def test_stft_domain_networks_run_in_float32():
    from espnet3.systems.snoar.src.net import build_net

    stft = build_net(
        {
            "encoder": "stft",
            "encoder_conf": {"n_fft": 64, "hop_length": 32},
            "decoder": "stft",
            "decoder_conf": {"n_fft": 64, "hop_length": 32},
            "separator": "tfgridnetv3",
            "separator_conf": {
                "n_srcs": 1,
                "n_layers": 1,
                "emb_dim": 4,
                "lstm_hidden_units": 8,
                "attn_n_head": 1,
            },
        }
    )
    assert stft.fp32 and not build_net(tiny_net(1)).fp32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = stft(torch.randn(1, 2000))
    assert out.dtype == torch.float32 and out.shape == (1, 1, 2000)


def test_stft_network_receives_the_sampling_rate():
    """An STFT model with ``default_fs: 8000`` must get fs to scale its window."""
    from espnet3.systems.snoar.src.net import build_net

    conf = {
        "encoder": "stft",
        "encoder_conf": {"n_fft": 64, "hop_length": 32, "default_fs": 8000},
        "decoder": "stft",
        "decoder_conf": {"n_fft": 64, "hop_length": 32, "default_fs": 8000},
        "separator": "tfgridnetv3",
        "separator_conf": {
            "n_srcs": 1,
            "n_layers": 1,
            "lstm_hidden_units": 8,
            "attn_n_head": 1,
            "emb_dim": 4,
            "emb_ks": 1,
            "emb_hs": 1,
        },
    }
    seen = []
    net = build_net(conf, fs=16000)
    net.enh.encoder.register_forward_hook(
        lambda m, a, k, o: seen.append(o[0].shape[-1]), with_kwargs=True
    )
    assert net(torch.randn(1, 1600)).shape == (1, 1, 1600)
    assert seen == [65]  # n_fft 64 * 16000 / 8000 = 128 -> 65 bins


def test_bsrnn_runs_under_bf16_autocast():
    """BSRNN follows the trainer's bf16 precision; its complex mask stays fp32."""
    from espnet3.systems.snoar.src.net import build_net

    stft = {"n_fft": 960, "hop_length": 480, "default_fs": 48000}
    conf = {
        "encoder": "stft",
        "encoder_conf": {**stft, "use_builtin_complex": True},
        "decoder": "stft",
        "decoder_conf": stft,
        "separator": "bsrnn",
        "separator_conf": {"num_spk": 2, "num_channels": 8, "num_layers": 1},
    }
    net = build_net(conf, fs=16000).eval()
    assert not net.fp32
    x = 0.1 * torch.randn(1, 3200)
    with torch.no_grad():
        ref = net(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = net(x).float()
    assert y.shape == (1, 2, 3200)
    assert ((y - ref).norm() / ref.norm()) < 0.05


def test_snoar_model_rejects_seed_without_mode_readout_slots():
    """A self-training mode without a seed is rejected (fs default unchanged)."""
    with pytest.raises(ValueError, match="needs a seed"):
        SNOARModel(mode="snoar", net={"separator": "rnn"})


def test_valid_utmos_reports_without_references(monkeypatch):
    import espnet3.systems.snoar.src.model as model_mod

    class FakeMOS(torch.nn.Module):
        def forward(self, wave, sr):
            assert sr == 16000 and wave.dtype == torch.float32
            return wave.abs().mean(-1)

    monkeypatch.setattr(model_mod, "_load_utmos", FakeMOS)
    m = SNOARModel("dnf", net=tiny_net(2), valid_utmos=True).eval()
    x = torch.randn(2, 3200)
    _, stats, _ = m(speech_mix=x, noise2=torch.randn(2, 3200))
    assert {"utmos", "utmos_delta"} <= set(stats)
    assert not any("_utmos" in k for k in m.state_dict())
