import pytest
import torch

from espnet2.tts.espnet_model import ESPnetTTSModel
from espnet3.systems.tts.f5_tts.builder import build_f5_tts_model
from espnet3.systems.tts.f5_tts.f5tts import F5TTS
from espnet3.systems.tts.f5_tts.vocoder_mel import VocoderMelSpec

TTS_CONF = dict(
    dim=32,
    depth=1,
    heads=2,
    dim_head=16,
    ff_mult=1,
    text_dim=16,
    conv_layers=1,
    odeint_method="euler",
)
FEATS_CONF = dict(
    fs=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=100,
    mel_spec_type="vocos",
)


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<blank>\n<unk>\na\nb\n<sos/eos>\n", encoding="utf-8")
    return str(path)


def test_builds_espnet_tts_model_with_f5_parts(token_file):
    model = build_f5_tts_model(
        token_list=token_file,
        feats_extract_conf=FEATS_CONF,
        tts_conf=TTS_CONF,
    )
    assert isinstance(model, ESPnetTTSModel)
    assert isinstance(model.feats_extract, VocoderMelSpec)
    assert isinstance(model.tts, F5TTS)
    assert model.normalize is None


def test_odim_is_derived_from_feats_extract(token_file):
    model = build_f5_tts_model(
        token_list=token_file,
        feats_extract_conf=FEATS_CONF,
        tts_conf=TTS_CONF,
    )
    assert model.feats_extract.output_size() == FEATS_CONF["n_mels"]
    assert model.tts.odim == FEATS_CONF["n_mels"]


def test_vocab_size_comes_from_the_token_file(token_file):
    model = build_f5_tts_model(
        token_list=token_file,
        feats_extract_conf=FEATS_CONF,
        tts_conf=TTS_CONF,
    )
    assert model.tts.cfm.transformer.text_embed.text_embed.num_embeddings == 5 + 1


def test_token_list_may_be_a_list():
    model = build_f5_tts_model(
        token_list=["<blank>", "<unk>", "a", "b", "<sos/eos>"],
        feats_extract_conf=FEATS_CONF,
        tts_conf=TTS_CONF,
    )
    assert isinstance(model, ESPnetTTSModel)


def test_rejects_bad_token_list():
    with pytest.raises(RuntimeError):
        build_f5_tts_model(
            token_list=42,
            feats_extract_conf=FEATS_CONF,
            tts_conf=TTS_CONF,
        )


def test_rejects_explicit_odim(token_file):
    with pytest.raises(RuntimeError):
        build_f5_tts_model(
            token_list=token_file,
            feats_extract_conf=FEATS_CONF,
            tts_conf=TTS_CONF,
            odim=100,
        )


def test_rejects_unknown_top_level_keys(token_file):
    """Dead espnet2-task keys must fail loudly, not be silently ignored."""
    with pytest.raises(TypeError):
        build_f5_tts_model(
            token_list=token_file,
            feats_extract_conf=FEATS_CONF,
            tts_conf=TTS_CONF,
            token_type="char",
        )


def test_rejects_typo_inside_tts_conf(token_file):
    """A misspelled hyper-parameter must not silently train a default model."""
    with pytest.raises(TypeError):
        build_f5_tts_model(
            token_list=token_file,
            feats_extract_conf=FEATS_CONF,
            tts_conf=dict(TTS_CONF, dpeth=18),
        )


def test_rejects_typo_inside_feats_extract_conf(token_file):
    with pytest.raises(TypeError):
        build_f5_tts_model(
            token_list=token_file,
            feats_extract_conf=dict(FEATS_CONF, n_mel=100),
            tts_conf=TTS_CONF,
        )


def test_tts_conf_scales_the_backbone(token_file):
    """`tts_conf` is the scaling knob: its keys reach F5TTS directly."""
    model = build_f5_tts_model(
        token_list=token_file,
        feats_extract_conf=FEATS_CONF,
        tts_conf=dict(TTS_CONF, dim=64, depth=3),
    )
    backbone = model.tts.cfm.transformer
    assert backbone.dim == 64
    assert backbone.depth == 3
    assert len(backbone.transformer_blocks) == 3
    assert backbone.transformer_blocks[0].attn.to_q.in_features == 64


def test_collect_feats_is_available(token_file):
    model = build_f5_tts_model(
        token_list=token_file,
        feats_extract_conf=FEATS_CONF,
        tts_conf=TTS_CONF,
    )
    out = model.collect_feats(
        text=torch.zeros(1, 4, dtype=torch.long),
        text_lengths=torch.tensor([4]),
        speech=torch.randn(1, 24000),
        speech_lengths=torch.tensor([24000]),
    )
    assert "feats" in out and "feats_lengths" in out


def test_accepts_omegaconf_containers(token_file):
    """Hydra passes nested blocks as DictConfig unless `_convert_` is set."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "espnet3.systems.tts.f5_tts.builder.build_f5_tts_model",
            "token_list": token_file,
            "feats_extract_conf": dict(FEATS_CONF),
            "tts_conf": dict(TTS_CONF, frac_lengths_mask=[0.7, 1.0]),
            "model_conf": {},
        }
    )
    model = instantiate(cfg)
    assert isinstance(model, ESPnetTTSModel)
    assert isinstance(model.tts.cfm.frac_lengths_mask, tuple)
