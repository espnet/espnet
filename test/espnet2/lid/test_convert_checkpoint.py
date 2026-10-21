"""Check original checkpoint renaming independently of model downloads."""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

PATH = (
    Path(__file__).resolve().parents[3] / "egs2/geolid/lid1/local/convert_checkpoint.py"
)
spec = importlib.util.spec_from_file_location("convert_checkpoint", PATH)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_conversion_strict_load(tiny_geolid):
    _, model, _ = tiny_geolid
    source = {}
    prefix = converter.PREFIX
    for key, value in model.state_dict().items():
        old = (
            ("loss.lang2vec_head.0." + key.removeprefix("lang2vec_head."))
            if key.startswith("lang2vec_head.")
            else key
        )
        for i in (0, 1):
            layer = prefix + f"layers.{i}."
            if key.startswith(layer):
                suffix = key.removeprefix(layer)
                module, parameter = suffix.split(".", 1)
                if module == "layer":
                    old = layer + parameter
                elif module == "conditioning_layer":
                    old = prefix + "lang2vec_conditioning_projs." + parameter
                else:
                    group = {
                        "encoder": "ecapa_encoder",
                        "prediction": "lang2vec_head",
                    }.get(module, module)
                    old = (
                        prefix
                        + f"{group}.{i}."
                        + ("0." if module == "prediction" else "")
                        + parameter
                    )
        source[old] = value
    source[prefix + "aamsoftmax_loss.weight"] = source["loss.weight"]
    converted = converter.convert_state_dict(source, [0, 1])
    model.load_state_dict(converted, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, converted[key])


def test_conflicting_alias_is_rejected():
    source = {
        "loss.weight": torch.ones(2),
        converter.PREFIX + "aamsoftmax_loss.weight": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="Conflicting checkpoint aliases"):
        converter.convert_state_dict(source, [32])


def test_early_independent_projection_names():
    weight = torch.randn(4, 3)
    converted = converter.convert_state_dict(
        {converter.PREFIX + "conditioning_projs.32.weight": weight}, [32]
    )
    assert torch.equal(
        converted[converter.PREFIX + "layers.32.conditioning_layer.weight"], weight
    )


def test_converted_config_standard_inference(tiny_geolid, tmp_path):
    import soundfile as sf

    from espnet2.bin.lid_inference import main

    args, model, base = tiny_geolid
    source = vars(args).copy()
    source["use_amp"] = True
    # Older training configurations may lack fields added to TrainerOptions.
    del source["ddp_comm_hook"]
    config = converter.convert_config(source)
    assert config["use_amp"] is True
    assert config["ddp_comm_hook"] is None
    config["frontend_conf"]["frontend_conf"]["path_or_url"] = str(base)
    labels = tmp_path / "languages.txt"
    labels.write_text("eng\nfra\nspa\nzho\n")
    config["lang2utt"] = str(labels)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)
    audio = tmp_path / "speech.wav"
    sf.write(audio, torch.randn(16000).numpy() * 0.01, 16000)
    wav_scp = tmp_path / "wav.scp"
    wav_scp.write_text(f"test_utterance {audio}\n")
    output = tmp_path / "output"
    output.mkdir()
    main(
        [
            "--output_dir",
            str(output),
            "--ngpu",
            "0",
            "--lid_train_config",
            str(config_path),
            "--lid_model_file",
            str(checkpoint),
            "--data_path_and_name_and_type",
            f"{wav_scp},speech,sound",
            "--use_preprocessor",
            "false",
            "--fix_duration",
            "false",
            "--num_workers",
            "0",
            "--extract_embd",
            "false",
            "--save_tsne_plot",
            "false",
        ]
    )
    result = (output / f"{tmp_path.name}_lids").read_text().split()
    assert result[0] == "test_utterance"
    assert result[1] in labels.read_text().splitlines()


def test_conversion_accepts_previous_stock_frontend_layout(tiny_geolid):
    _, model, _ = tiny_geolid
    source = {
        (
            ("loss.lang2vec_head.0." + k.removeprefix("lang2vec_head."))
            if k.startswith("lang2vec_head.")
            else k
        ): v
        for k, v in model.state_dict().items()
    }
    model.load_state_dict(converter.convert_state_dict(source, [0, 1]), strict=True)


def test_legacy_config_moves_auxiliary_options(tiny_geolid):
    args, _, _ = tiny_geolid
    source = vars(args).copy()
    source["model"] = "upstream_condition"
    source["loss"] = "aamsoftmax_sc_topk_lang2vec"
    source["model_conf"] = dict(source["model_conf"])
    for key in ("lang2vec_dim", "lang2vec_weight", "lang2vec_loss"):
        source["model_conf"].pop(key)
    source["loss_conf"] = dict(
        k_top=2, lang2vec_dim=3, lang2vec_weight=0.2, lang2vec_type="syntax_knn"
    )
    config = converter.convert_config(source)
    assert config["model"] == "lang2vec"
    assert config["loss"] == "aamsoftmax_sc_topk"
    assert config["loss_conf"] == {"k_top": 2}
    assert config["model_conf"]["lang2vec_dim"] == 3
    assert config["model_conf"]["lang2vec_loss"] == "bce"
    assert config["model_conf"]["lang2vec_weight"] == 0.2
