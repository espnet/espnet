import json
from test.espnet2.universa.test_audio_metric_task import task_args

import numpy as np
import pytest
import soundfile
import torch
import yaml

from espnet2.bin.audio_metric_inference import UniversaInference, inference
from espnet2.tasks.audio_metric import AudioMetricTask


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
    wav = tmp_path / "audio.wav"
    soundfile.write(wav, np.random.randn(256).astype(np.float32), 16000)
    scp = tmp_path / "wav.scp"
    scp.write_text(f"a {wav}\n")
    inference(
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
    key, scores = (tmp_path / "output/metric.scp").read_text().split(maxsplit=1)
    assert key == "a"
    assert set(json.loads(scores)) == expected
    if multi_branch == "ar":
        assert (tmp_path / "output/token.scp").read_text().startswith("a 2 ")
