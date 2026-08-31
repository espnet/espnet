import pytest
import torch

from espnet3.systems.tts.f5_tts.f5tts import F5TTS
from espnet3.systems.tts.f5_tts.vocoder_mel import VocoderMelSpec

MODEL_CONF = dict(
    hidden_size=32,
    depth=1,
    attention_heads=2,
    attention_head_size=16,
    feed_forward_multiplier=1,
    text_embedding_size=16,
    convolution_layers=1,
    ode_solver_method="euler",
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


def test_model_is_a_plain_module_owning_its_front_end(token_file):
    """ESPnet3 needs only an nn.Module; the mel front end lives in the model."""
    model = F5TTS(token_list=token_file, feats_extract_config=FEATS_CONF, **MODEL_CONF)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.feats_extract, VocoderMelSpec)


def test_mel_dim_is_derived_from_feats_extract(token_file):
    model = F5TTS(token_list=token_file, feats_extract_config=FEATS_CONF, **MODEL_CONF)
    assert model.feats_extract.output_size == FEATS_CONF["n_mels"]
    assert model.mel_dim == FEATS_CONF["n_mels"]


def test_vocab_size_comes_from_the_token_file(token_file):
    model = F5TTS(token_list=token_file, feats_extract_config=FEATS_CONF, **MODEL_CONF)
    assert model.cfm.transformer.text_embed.text_embed.num_embeddings == 5 + 1


def test_token_list_may_be_a_list():
    model = F5TTS(
        token_list=["<blank>", "<unk>", "a", "b", "<sos/eos>"],
        feats_extract_config=FEATS_CONF,
        **MODEL_CONF,
    )
    assert model.cfm.transformer.text_embed.text_embed.num_embeddings == 5 + 1


def test_rejects_bad_token_list():
    with pytest.raises(RuntimeError):
        F5TTS(token_list=42, feats_extract_config=FEATS_CONF, **MODEL_CONF)


def test_rejects_unknown_top_level_keys(token_file):
    """Dead espnet2-task keys must fail loudly, not be silently ignored."""
    with pytest.raises(TypeError):
        F5TTS(
            token_list=token_file,
            feats_extract_config=FEATS_CONF,
            token_type="char",
            **MODEL_CONF,
        )


def test_rejects_typo_in_a_hyper_parameter(token_file):
    """A misspelled hyper-parameter must not silently train a default model."""
    with pytest.raises(TypeError):
        F5TTS(
            token_list=token_file,
            feats_extract_config=FEATS_CONF,
            **dict(MODEL_CONF, dpeth=18),
        )


def test_rejects_typo_inside_feats_extract_config(token_file):
    with pytest.raises(TypeError):
        F5TTS(
            token_list=token_file,
            feats_extract_config=dict(FEATS_CONF, n_mel=100),
            **MODEL_CONF,
        )


def test_hyper_parameters_scale_the_backbone(token_file):
    model = F5TTS(
        token_list=token_file,
        feats_extract_config=FEATS_CONF,
        **dict(MODEL_CONF, hidden_size=64, depth=3),
    )
    backbone = model.cfm.transformer
    assert backbone.dim == 64
    assert backbone.depth == 3
    assert len(backbone.transformer_blocks) == 3
    assert backbone.transformer_blocks[0].attn.to_q.in_features == 64


def test_collect_feats_is_available(token_file):
    model = F5TTS(token_list=token_file, feats_extract_config=FEATS_CONF, **MODEL_CONF)
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

    config = OmegaConf.create(
        {
            "_target_": "espnet3.systems.tts.f5_tts.f5tts.F5TTS",
            "token_list": token_file,
            "feats_extract_config": dict(FEATS_CONF),
            **dict(MODEL_CONF, mask_fraction_range=[0.7, 1.0]),
        }
    )
    model = instantiate(config)
    assert isinstance(model, F5TTS)
    assert isinstance(model.cfm.frac_lengths_mask, tuple)
