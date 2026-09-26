from espnet2.tasks.audio_metric import AudioMetricTask


def task_args(tmp_path):
    metrics = tmp_path / "metric2id"
    metrics.write_text("mos\nwer\n")
    args = AudioMetricTask.get_parser().parse_args(
        [
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
    from espnet2.tasks.universa import UniversaTask

    assert UniversaTask is AudioMetricTask
    args = task_args(tmp_path)
    model = AudioMetricTask.build_model(args)
    assert model.universa.metric2id == {"mos": 0, "wer": 1}
    assert args.metric2id == ["mos", "wer"]
