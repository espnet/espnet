from argparse import Namespace

import pytest
import torch
import yaml

from espnet2.bin.rst_inference import _load_feature_predictor
from espnet2.rst.rst_model import W2VBert2Encoder, merge_lora_adapters
from espnet2.tasks.rst import RestorationTask
from espnet2.tasks.rst_vocoder import RestorationVocoderTask

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


@pytest.fixture(scope="module")
def tiny_w2v_bert(tmp_path_factory):
    """A randomly initialised 2-layer w2v-BERT 2.0 in Hugging Face format."""
    torch.manual_seed(0)
    config = transformers.Wav2Vec2BertConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_depthwise_kernel_size=3,
        apply_spec_augment=False,
    )
    path = tmp_path_factory.mktemp("tiny_w2v_bert")
    transformers.Wav2Vec2BertModel(config).save_pretrained(path)
    return str(path)


def _randomise_lora_b(module):
    # LoRA B starts at zero, which would make any merge look exact
    generator = torch.Generator().manual_seed(0)
    for name, param in module.named_parameters():
        if "lora_B" in name:
            param.data.normal_(std=0.1, generator=generator)


def _encoder(tag):
    encoder = W2VBert2Encoder(
        model_tag=tag, target_layer=2, lora_rank=4, lora_alpha=8, lora_dropout=0.0
    )
    return encoder.eval()


def _features(encoder):
    wav = 0.1 * torch.randn(2, 16000, generator=torch.Generator().manual_seed(1))
    inputs = encoder._wav_to_ssl_inputs(wav, torch.tensor([16000, 12000]))
    with torch.no_grad():
        return encoder.encode(inputs)[0]


def _has_lora(module):
    return any("lora_" in key for key in module.state_dict())


def test_merge_preserves_features(tiny_w2v_bert):
    encoder = _encoder(tiny_w2v_bert)
    _randomise_lora_b(encoder.student)
    adapted = _features(encoder)
    # two feed-forward modules (output_dense) per layer
    assert merge_lora_adapters(encoder.student) == 4
    assert not _has_lora(encoder)
    torch.testing.assert_close(_features(encoder), adapted, atol=1e-5, rtol=1e-4)
    assert merge_lora_adapters(encoder.student) == 0


def test_inference_loader_merges(tiny_w2v_bert, tmp_path):
    train_config = dict(
        ssl_encoder="w2v_bert2",
        ssl_encoder_conf=dict(model_tag=tiny_w2v_bert, target_layer=2),
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        input_sr=16000,
    )
    model = RestorationTask.build_model(Namespace(**train_config)).eval()
    _randomise_lora_b(model)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(train_config))
    model_path = tmp_path / "valid.loss.best.pth"
    torch.save(model.state_dict(), model_path)

    loaded = _load_feature_predictor(str(config_path), str(model_path), "cpu")
    assert not _has_lora(loaded)
    torch.testing.assert_close(
        _features(loaded.ssl_encoder),
        _features(model.ssl_encoder),
        atol=1e-5,
        rtol=1e-4,
    )


def test_vocoder_loader_merges(tiny_w2v_bert, tmp_path):
    adapted = _encoder(tiny_w2v_bert)
    _randomise_lora_b(adapted.student)
    path = tmp_path / "valid.loss.best.pth"
    torch.save({f"ssl_encoder.{k}": v for k, v in adapted.state_dict().items()}, path)

    encoder = _encoder(tiny_w2v_bert)
    RestorationVocoderTask._load_feature_predictor(encoder, str(path))
    assert not _has_lora(encoder)
    torch.testing.assert_close(
        _features(encoder), _features(adapted), atol=1e-5, rtol=1e-4
    )

    # a checkpoint without the adapter still cannot pass for a trained predictor
    merge_lora_adapters(adapted.student)
    plain = tmp_path / "no_adapter.pth"
    torch.save({f"ssl_encoder.{k}": v for k, v in adapted.state_dict().items()}, plain)
    with pytest.raises(RuntimeError, match="LoRA adapter"):
        RestorationVocoderTask._load_feature_predictor(
            _encoder(tiny_w2v_bert), str(plain)
        )
