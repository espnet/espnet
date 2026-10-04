import importlib
import json
from test.espnet2.universa.test_metric_tokenizer import token_info

import pytest
import torch

from espnet2.tasks.audio_metric import AudioMetricTask
from espnet2.torch_utils.initialize import INITIALIZATIONS
from espnet2.train.collate_fn import UniversaCollateFn
from espnet2.train.preprocessor import UniversaProcessor
from espnet2.universa.ar_universa.data import ARMetricCollateFn, ARMetricProcessor


def task_args(tmp_path, *options):
    """Parse CLI options and use a small predictor for task tests."""
    metrics = tmp_path / "metric2id"
    metrics.write_text("mos\nwer\n")
    args = AudioMetricTask.get_parser().parse_args(
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


def test_legacy_task_and_portable_vocabulary(tmp_path):
    """Keep historical imports and embed metric names for checkpoint reloads."""
    from espnet2.tasks.universa import UniversaTask

    assert UniversaTask is AudioMetricTask
    args = task_args(tmp_path)
    model = AudioMetricTask.build_model(args)
    assert model.universa.metric2id == {"mos": 0, "wer": 1}
    assert args.metric2id == ["mos", "wer"]


@pytest.mark.parametrize("init", [*INITIALIZATIONS, "none"])
def test_supported_initializations(tmp_path, init):
    """Every advertised initialization can construct a model."""
    args = task_args(tmp_path, "--init", init)
    model = AudioMetricTask.build_model(args)
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
            AudioMetricTask.build_model(args)
        assert args.sequential_metric is False
        return
    model = AudioMetricTask.build_model(args)
    assert model.universa.sequential_metrics is True
    assert args.sequential_metric is True
    assert args.metric2type == {"mos": "numerical", "language": "categorical"}
    assert args.metric_token_info == token_info()
    types.unlink()
    info.unlink()
    rebuilt = AudioMetricTask.build_model(args)
    assert rebuilt.universa.metric2id == model.universa.metric2id
    assert isinstance(
        AudioMetricTask.build_preprocess_fn(args, True), ARMetricProcessor
    )
    assert isinstance(AudioMetricTask.build_collate_fn(args, True), ARMetricCollateFn)


def test_base_data_setup(tmp_path):
    """Build scalar-metric data helpers before and after metadata normalization."""
    args = task_args(tmp_path)
    assert isinstance(AudioMetricTask.build_collate_fn(args, True), UniversaCollateFn)
    AudioMetricTask.build_model(args)
    assert isinstance(AudioMetricTask.build_collate_fn(args, False), UniversaCollateFn)
    assert isinstance(
        AudioMetricTask.build_preprocess_fn(args, True), UniversaProcessor
    )
    args.use_preprocessor = False
    assert AudioMetricTask.build_preprocess_fn(args, False) is None
    for inference in (False, True):
        assert AudioMetricTask.required_data_names(inference=inference) == (
            ("audio",) if inference else ("metrics", "audio")
        )
        assert AudioMetricTask.optional_data_names(inference=inference) == (
            "ref_audio",
            "ref_text",
        )


@pytest.mark.parametrize("entry", ["audio_metric_train", "universa_train"])
def test_training_entry_points(entry):
    """Both entry points expose a usable parser and print task configuration."""
    module = importlib.import_module("espnet2.bin." + entry)
    assert module.get_parser().parse_args([]).universa == "base"
    with pytest.raises(SystemExit) as exc:
        module.main(["--print_config"])
    assert exc.value.code == 0
