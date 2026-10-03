from argparse import ArgumentParser, Namespace

import pytest
import torch
import yaml

from espnet2.bin.rst_inference import _load_feature_predictor
from espnet2.bin.rst_merge_lora import get_parser, main
from espnet2.tasks.rst import RestorationTask

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


@pytest.fixture
def predictor(tmp_path):
    """A trained-looking predictor: tiny w2v-BERT 2.0 with a non-zero adapter."""
    torch.manual_seed(0)
    config = transformers.Wav2Vec2BertConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_depthwise_kernel_size=3,
        apply_spec_augment=False,
    )
    tag = tmp_path / "tiny_w2v_bert"
    transformers.Wav2Vec2BertModel(config).save_pretrained(tag)

    train_config = dict(
        ssl_encoder="w2v_bert2",
        ssl_encoder_conf=dict(model_tag=str(tag), target_layer=2),
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        input_sr=16000,
    )
    model = RestorationTask.build_model(Namespace(**train_config))
    generator = torch.Generator().manual_seed(0)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(std=0.1, generator=generator)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(train_config))
    model_path = tmp_path / "valid.loss.best.pth"
    torch.save(model.state_dict(), model_path)
    return str(config_path), str(model_path)


def _features(model):
    encoder = model.ssl_encoder
    wav = 0.1 * torch.randn(1, 16000, generator=torch.Generator().manual_seed(1))
    inputs = encoder._wav_to_ssl_inputs(wav, torch.tensor([16000]))
    with torch.no_grad():
        return encoder.encode(inputs)[0]


def test_main_writes_an_equivalent_merged_predictor(predictor, tmp_path):
    config_path, model_path = predictor
    out = tmp_path / "merged"
    main(
        [
            "--train_config",
            config_path,
            "--model_file",
            model_path,
            "--output_dir",
            str(out),
        ]
    )
    merged_config = yaml.safe_load((out / "config.yaml").read_text())
    assert merged_config["lora_rank"] == 0
    merged = _load_feature_predictor(
        str(out / "config.yaml"), str(out / "model.pth"), "cpu"
    )
    assert not any("lora_" in key for key in merged.state_dict())
    adapted = _load_feature_predictor(config_path, model_path, "cpu")
    torch.testing.assert_close(
        _features(merged), _features(adapted), atol=1e-5, rtol=1e-4
    )


def test_main_refuses_a_predictor_without_adapter(predictor, tmp_path):
    config_path, model_path = predictor
    out = tmp_path / "merged"
    main(
        [
            "--train_config",
            config_path,
            "--model_file",
            model_path,
            "--output_dir",
            str(out),
            "--check_duration",
            "0",
        ]
    )
    with pytest.raises(RuntimeError, match="no LoRA adapter"):
        main(
            [
                "--train_config",
                str(out / "config.yaml"),
                "--model_file",
                str(out / "model.pth"),
                "--output_dir",
                str(tmp_path / "again"),
            ]
        )
