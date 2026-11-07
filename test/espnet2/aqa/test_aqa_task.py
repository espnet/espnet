import importlib
import json
from test.espnet2.aqa.test_metric_tokenizer import token_info

import numpy as np
import pytest
import soundfile
import torch
import yaml

from espnet2.aqa.ar_universa.data import ARMetricCollateFn, ARMetricProcessor
from espnet2.bin.aqa_inference import main as inference_main
from espnet2.tasks.aqa import AqaTask
from espnet2.torch_utils.initialize import INITIALIZATIONS
from espnet2.train.collate_fn import UniversaCollateFn
from espnet2.train.preprocessor import UniversaProcessor


def task_args(tmp_path, *options):
    """Parse CLI options and use a small predictor for task tests."""
    metrics = tmp_path / "metric2id"
    metrics.write_text("mos\nwer\n")
    args = AqaTask.get_parser().parse_args(
        [
            *options,
            "--metric2id",
            str(metrics),
            "--use_ref_text",
            "false",
            "--use_ref_audio",
            "false",
        ]
    )
    args.frontend_conf = dict(n_fft=64, hop_length=16, n_mels=8)
    args.universa_conf = dict(
        embedding_size=8,
        audio_encoder_params=dict(
            num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
        ),
        cross_attention_params=dict(n_head=2, dropout_rate=0.0),
    )
    return args


def test_portable_vocabulary(tmp_path):
    """Embed metric names for checkpoint reloads without the training directory."""
    args = task_args(tmp_path)
    model = AqaTask.build_model(args)
    assert model.universa.metric2id == {"mos": 0, "wer": 1}
    assert args.metric2id == ["mos", "wer"]


@pytest.mark.parametrize("init", [*INITIALIZATIONS, "none"])
def test_supported_initializations(tmp_path, init):
    """Every advertised initialization can construct a model."""
    args = task_args(tmp_path, "--init", init)
    model = AqaTask.build_model(args)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert args.sequential_metric is False


def test_reject_unsupported_initialization(tmp_path):
    """Reject obsolete initialization names during argument parsing."""
    with pytest.raises(SystemExit):
        task_args(tmp_path, "--init", "chainer")


@pytest.mark.parametrize("sequential", [None, True, False])
def test_arecho_sequential_default_and_portable_metadata(tmp_path, sequential):
    """Default ARECHO to sequential targets while respecting explicit values."""
    options = [] if sequential is None else ["--sequential_metric", str(sequential)]
    args = task_args(tmp_path, "--universa", "ar_universa", *options)
    args.metric2id = ["mos", "language"]
    types = tmp_path / "metric2type"
    types.write_text("mos numerical\nlanguage categorical\n")
    info = tmp_path / "tokens.json"
    info.write_text(json.dumps(token_info()))
    args.metric2type, args.metric_token_info = str(types), str(info)
    args.universa_conf.update(
        embedding_dim=256,
        metric_decoder_params=dict(num_blocks=1, attention_heads=2, linear_units=16),
    )
    if sequential is False:
        with pytest.raises(ValueError, match="sequential_metrics is required"):
            AqaTask.build_model(args)
        assert args.sequential_metric is False
        return
    model = AqaTask.build_model(args)
    assert model.universa.sequential_metrics is True
    assert args.sequential_metric is True
    assert args.metric2type == {"mos": "numerical", "language": "categorical"}
    assert args.metric_token_info == token_info()
    types.unlink()
    info.unlink()
    rebuilt = AqaTask.build_model(args)
    assert rebuilt.universa.metric2id == model.universa.metric2id
    assert isinstance(AqaTask.build_preprocess_fn(args, True), ARMetricProcessor)
    assert isinstance(AqaTask.build_collate_fn(args, True), ARMetricCollateFn)


@pytest.mark.parametrize("option", [None, "false", "true"])
def test_defer_full_meta_reaches_collate(tmp_path, option):
    """The training option orders train and valid targets; old configs keep it off."""
    options = [] if option is None else ["--defer_full_meta", option]
    args = task_args(tmp_path, "--universa", "ar_universa", *options)
    assert args.defer_full_meta is (option == "true")
    for train in (True, False):
        collate = AqaTask.build_collate_fn(args, train)
        assert collate.defer_full_meta is (option == "true")
        assert collate.randomize is train
    del args.defer_full_meta  # A config saved before the option existed.
    assert AqaTask.build_collate_fn(args, True).defer_full_meta is False


@pytest.mark.parametrize(
    "trained, option, last",
    [
        (None, [], False),  # A config saved before the option existed.
        (False, [], False),
        (True, [], True),
        (True, ["--defer_full_meta", "false"], False),
        (False, ["--defer_full_meta", "true"], True),
    ],
)
def test_inference_defer_full_meta_follows_training(
    tmp_path, full_token_info, trained, option, last
):
    """Inference decodes whole-clip metrics last as trained unless overridden."""
    args = task_args(tmp_path, "--universa", "ar_universa")
    args.sequential_metric = True
    args.metric2id = ["mos_full", "mos", "language"]
    args.metric2type = dict(
        mos_full="numerical", mos="numerical", language="categorical"
    )
    args.metric_token_info = full_token_info
    args.universa_conf["metric_decoder_params"] = dict(
        num_blocks=1, attention_heads=2, linear_units=16
    )
    if trained is None:
        del args.defer_full_meta
    else:
        args.defer_full_meta = trained
    model = AqaTask.build_model(args)
    with torch.no_grad():
        model.universa.decoder.output_layer.bias[14] = 100  # Favor mos_full.
    config, checkpoint = tmp_path / "config.yaml", tmp_path / "model.pth"
    config.write_text(yaml.safe_dump(vars(args)))
    torch.save(model.state_dict(), checkpoint)
    wav, scp = tmp_path / "audio.wav", tmp_path / "wav.scp"
    soundfile.write(wav, np.random.randn(256).astype(np.float32), 16000)
    scp.write_text(f"a {wav}\n")
    output = tmp_path / "output"
    inference_main(
        [
            "--output_dir",
            str(output),
            "--train_config",
            str(config),
            "--model_file",
            str(checkpoint),
            "--data_path_and_name_and_type",
            f"{scp},audio,sound",
            "--save_token_seq",
            "true",
            "--num_workers",
            "0",
            *option,
        ]
    )
    labels = (output / "token.scp").read_text().split()[2::2]
    assert labels.index("14") == (2 if last else 0)


def test_base_data_setup(tmp_path):
    """Build scalar-metric data helpers before and after metadata normalization."""
    args = task_args(tmp_path)
    assert isinstance(AqaTask.build_collate_fn(args, True), UniversaCollateFn)
    AqaTask.build_model(args)
    assert isinstance(AqaTask.build_collate_fn(args, False), UniversaCollateFn)
    assert isinstance(AqaTask.build_preprocess_fn(args, True), UniversaProcessor)
    args.use_preprocessor = False
    assert AqaTask.build_preprocess_fn(args, False) is None
    for inference in (False, True):
        assert AqaTask.required_data_names(inference=inference) == (
            ("audio",) if inference else ("metrics", "audio")
        )
        assert AqaTask.optional_data_names(inference=inference) == (
            "ref_audio",
            "ref_text",
        )


def test_training_entry_point():
    """The AQA entry point exposes a parser and prints task configuration."""
    module = importlib.import_module("espnet2.bin.aqa_train")
    assert module.get_parser().parse_args([]).universa == "base"
    with pytest.raises(SystemExit) as exc:
        module.main(["--print_config"])
    assert exc.value.code == 0
