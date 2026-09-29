import json
from test.espnet2.universa.test_audio_metric_task import task_args

import numpy as np
import pytest
import soundfile
import torch
import yaml

from espnet2.bin.audio_metric_inference import UniversaInference, inference
from espnet2.tasks.audio_metric import AudioMetricTask


@pytest.mark.parametrize("use_ref_text", [False, True])
def test_string_reference_text(tmp_path, use_ref_text):
    """Tokenize supported references and clearly reject unsupported strings."""
    args = task_args(tmp_path)
    args.use_ref_text = use_ref_text
    args.token_type = "char"
    args.token_list = ["<blank>", "<unk>", "a", "b", "<sos/eos>"]
    args.universa_conf["text_encoder_params"] = dict(
        num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
    )
    model = AudioMetricTask.build_model(args)
    config, checkpoint = tmp_path / "config.yaml", tmp_path / "model.pth"
    config.write_text(yaml.safe_dump(vars(args)))
    torch.save(model.state_dict(), checkpoint)
    predict = UniversaInference(config, checkpoint)
    audio = torch.randn(256)
    if not use_ref_text:
        with pytest.raises(
            ValueError, match="String ref_text requires use_ref_text=True"
        ):
            predict(audio, ref_text="ab")
        return
    string_result = predict(audio, ref_text="ab")
    token_result = predict(audio, ref_text=torch.tensor([2, 3]))
    for metric in ("mos", "wer"):
        assert np.isfinite(string_result[metric]).all()
        np.testing.assert_allclose(string_result[metric], token_result[metric])


def test_legacy_cli():
    from espnet2.bin import universa_inference

    assert universa_inference.UniversaInference is UniversaInference
    with pytest.raises(SystemExit) as exc:
        universa_inference.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("multi_branch", [False, True, "ar"])
def test_checkpoint_and_cli_inference(tmp_path, multi_branch):
    args = task_args(tmp_path)
    if multi_branch == "ar":
        from test.espnet2.universa.test_metric_tokenizer import token_info

        args.universa = "ar_universa"
        args.sequential_metric = True
        args.metric2id = ["mos", "language"]
        args.metric2type = {"mos": "numerical", "language": "categorical"}
        args.metric_token_info = token_info()
        args.universa_conf["metric_decoder_params"] = dict(
            num_blocks=1, attention_heads=2, linear_units=16
        )
    else:
        args.universa_conf["multi_branch"] = multi_branch
    model = AudioMetricTask.build_model(args)
    config, checkpoint = tmp_path / "config.yaml", tmp_path / "model.pth"
    config.write_text(yaml.safe_dump(vars(args)))
    torch.save(model.state_dict(), checkpoint)
    # Inference must not depend on the training directory's metric vocabulary.
    (tmp_path / "metric2id").unlink()
    predict = UniversaInference(
        config,
        checkpoint,
        save_token_seq=True,
        use_fixed_order=True,
        fixed_metric_name_order="language,mos" if multi_branch == "ar" else "",
    )
    result = predict(torch.randn(256))
    expected = {"mos", "language"} if multi_branch == "ar" else {"mos", "wer"}
    assert expected.issubset(result)
    if multi_branch == "ar":
        assert result["token_seq"][0][1::2] == [10, 4]
        with pytest.raises(ValueError, match="ARECHO inference requires batch_size=1"):
            predict(torch.randn(2, 256))
    else:
        config.write_text(yaml.safe_dump({**vars(args), "use_preprocessor": False}))
        no_preprocess = UniversaInference(config, checkpoint)
        with pytest.raises(ValueError, match="provide token IDs"):
            no_preprocess(torch.randn(256), ref_text="reference")
        config.write_text(yaml.safe_dump(vars(args)))
    wav = tmp_path / "audio.wav"
    soundfile.write(wav, np.random.randn(256).astype(np.float32), 16000)
    scp = tmp_path / "wav.scp"
    scp.write_text(f"a {wav}\n")
    options = dict(
        output_dir=tmp_path / "output",
        batch_size=1,
        dtype="float32",
        ngpu=0,
        seed=0,
        num_workers=0,
        log_level="WARNING",
        data_path_and_name_and_type=[(str(scp), "audio", "sound")],
        key_file=None,
        train_config=str(config),
        model_file=str(checkpoint),
        model_tag=None,
        always_fix_seed=False,
        allow_variable_data_keys=False,
        save_token_seq=True,
    )
    if multi_branch == "ar":
        with pytest.raises(ValueError, match="ARECHO inference requires batch_size=1"):
            inference(**{**options, "batch_size": 2})
    inference(**options)
    key, scores = (tmp_path / "output/metric.scp").read_text().split(maxsplit=1)
    assert key == "a"
    assert set(json.loads(scores)) == expected
    if multi_branch == "ar":
        assert (tmp_path / "output/token.scp").read_text().startswith("a 2 ")
