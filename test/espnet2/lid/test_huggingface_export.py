import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
def test_export_matches_espnet(tiny_geolid, tmp_path, shared, legacy):
    from espnet2.tasks.lid import LIDTask

    args, model, base = tiny_geolid
    args.model_conf["shared_conditioning_proj"] = shared
    args.model_conf["legacy_inference"] = legacy
    if not shared or legacy:
        model = LIDTask.build_model(args)
    model.eval()
    root = Path(__file__).resolve().parents[3]
    script = root / "egs2/geolid/lid1/local/export_huggingface.py"
    spec = importlib.util.spec_from_file_location("export_geolid", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    checkpoint = tmp_path / "model.pth"
    config_path = tmp_path / "train.yaml"
    torch.save(model.state_dict(), checkpoint)
    config_path.write_text(yaml.safe_dump(vars(args)))
    output = tmp_path / "exported"
    exported = exporter.export(checkpoint, config_path, output, base)
    loaded, info = transformers.AutoModelForAudioClassification.from_pretrained(
        output,
        trust_remote_code=True,
        output_loading_info=True,
        local_files_only=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not loaded.config.label_mapping_verified
    assert loaded.config.id2label[0] == "LABEL_0"
    extractor = transformers.AutoFeatureExtractor.from_pretrained(output)
    audio = [torch.randn(880).numpy(), torch.randn(810).numpy()]
    inputs = extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
    raw = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(x) for x in audio], batch_first=True
    )
    with torch.no_grad():
        expected, prediction = model(raw, torch.tensor([880, 810]), extract_embd=True)
        result = loaded(**inputs)
        torch.testing.assert_close(result.embeddings, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(result.logits.argmax(-1), prediction)
        torch.testing.assert_close(result.logits, exported(**inputs).logits)
    # The exported source must not require either private fork or ESPnet imports.
    for source in output.glob("*.py"):
        assert "from espnet" not in source.read_text()
        assert "import s3prl" not in source.read_text()
