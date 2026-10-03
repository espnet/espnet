import pytest
import torch

from espnet2.rst.rst_model import W2VBert2Encoder, merge_lora_adapters
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


def _encoder(tag, lora_rank):
    return W2VBert2Encoder(
        model_tag=tag,
        target_layer=2,
        lora_rank=lora_rank,
        lora_alpha=8,
        lora_dropout=0.0,
    ).eval()


def _trained_encoder(tag):
    encoder = _encoder(tag, lora_rank=4)
    # LoRA B starts at zero, which would make any merge look exact
    generator = torch.Generator().manual_seed(0)
    for name, param in encoder.student.named_parameters():
        if "lora_B" in name:
            param.data.normal_(std=0.1, generator=generator)
    return encoder


def _features(encoder):
    wav = 0.1 * torch.randn(2, 16000, generator=torch.Generator().manual_seed(1))
    inputs = encoder._wav_to_ssl_inputs(wav, torch.tensor([16000, 12000]))
    with torch.no_grad():
        return encoder.encode(inputs)[0]


def test_merge_preserves_features(tiny_w2v_bert):
    encoder = _trained_encoder(tiny_w2v_bert)
    adapted = _features(encoder)
    # two feed-forward modules (output_dense) per layer
    assert merge_lora_adapters(encoder.student) == 4
    assert not any("lora_" in key for key in encoder.state_dict())
    torch.testing.assert_close(_features(encoder), adapted, atol=1e-5, rtol=1e-4)
    assert merge_lora_adapters(encoder.student) == 0


def test_merged_weights_load_without_adapter(tiny_w2v_bert):
    encoder = _trained_encoder(tiny_w2v_bert)
    adapted = _features(encoder)
    merge_lora_adapters(encoder.student)
    plain = _encoder(tiny_w2v_bert, lora_rank=0)
    assert not any("lora_" in key for key in plain.state_dict())
    plain.load_state_dict(encoder.state_dict(), strict=True)
    torch.testing.assert_close(_features(plain), adapted, atol=1e-5, rtol=1e-4)


def _checkpoint(encoder, path):
    state = {f"ssl_encoder.{k}": v for k, v in encoder.state_dict().items()}
    torch.save(state, path)
    return str(path)


def test_vocoder_loads_merged_predictor(tiny_w2v_bert, tmp_path):
    adapted = _trained_encoder(tiny_w2v_bert)
    unmerged = _checkpoint(adapted, tmp_path / "unmerged.pth")
    expected = _features(adapted)
    merge_lora_adapters(adapted.student)
    merged = _checkpoint(adapted, tmp_path / "merged.pth")

    plain = _encoder(tiny_w2v_bert, lora_rank=0)
    RestorationVocoderTask._load_feature_predictor(plain, merged)
    torch.testing.assert_close(_features(plain), expected, atol=1e-5, rtol=1e-4)

    # an adapter checkpoint must not be half-loaded into a merged layout, and a
    # merged checkpoint cannot satisfy an encoder that expects an adapter
    with pytest.raises(RuntimeError, match="not a merged predictor"):
        RestorationVocoderTask._load_feature_predictor(
            _encoder(tiny_w2v_bert, lora_rank=0), unmerged
        )
    with pytest.raises(RuntimeError, match="LoRA adapter"):
        RestorationVocoderTask._load_feature_predictor(
            _encoder(tiny_w2v_bert, lora_rank=4), merged
        )
