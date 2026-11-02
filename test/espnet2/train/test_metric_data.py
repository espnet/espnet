import numpy as np
import pytest
import soundfile
import torch

from espnet2.train.collate_fn import MappingCollateFn, UniversaCollateFn
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.iterable_dataset import IterableESPnetDataset
from espnet2.train.preprocessor import UniversaProcessor


@pytest.mark.parametrize("streaming", [False, True])
def test_metric_and_missing_reference_data(tmp_path, streaming):
    wav = tmp_path / "audio.wav"
    soundfile.write(wav, np.random.randn(256).astype(np.float32), 16000)
    (tmp_path / "wav.scp").write_text(f"a {wav}\n")
    (tmp_path / "ref.scp").write_text("a None\n")
    (tmp_path / "metric.scp").write_text('a {"mos": 2.5}\n')
    paths = [
        (str(tmp_path / "wav.scp"), "audio", "sound"),
        (str(tmp_path / "ref.scp"), "ref_audio", "sound"),
        (str(tmp_path / "metric.scp"), "metrics", "metric"),
    ]
    cls = IterableESPnetDataset if streaming else ESPnetDataset
    dataset = cls(paths, preprocess=UniversaProcessor(train=False))
    sample = next(iter(dataset)) if streaming else dataset["a"]
    collate = UniversaCollateFn(["mos", "wer"], metric_pad_value=-100)
    assert isinstance(repr(collate), str)
    keys, batch = collate([sample])
    assert keys == ["a"]
    assert batch["metrics"]["mos"].item() == 2.5
    assert batch["metrics"]["wer"].item() == -100
    assert torch.count_nonzero(batch["ref_audio"]) == 0


@pytest.mark.parametrize("name", ["metrics", "annotations"])
@pytest.mark.parametrize("structured", [False, True])
def test_stats_preserve_tensor_metrics(tmp_path, structured, name):
    from espnet2.main_funcs.collect_stats import collect_stats

    batch = {"audio": torch.ones(2, 8), "audio_lengths": torch.tensor([8, 5])}
    if structured:
        batch[name] = {"mos": torch.tensor([2.0, 3.0])}
    else:
        batch[name] = torch.ones(2, 4)
        batch[name + "_lengths"] = torch.tensor([4, 2])
    iterator = [(["a", "b"], batch)]
    collect_stats(None, iterator, iterator, tmp_path, 0, 10, False)
    for mode in ("train", "valid"):
        directory = tmp_path / mode
        assert (directory / "audio_shape").read_text() == "a 8\nb 5\n"
        keys = (directory / "batch_keys").read_text().splitlines()
        assert keys == (["audio"] if structured else ["audio", name])
        if structured:
            assert not (directory / f"{name}_shape").exists()
        else:
            assert (directory / f"{name}_shape").read_text() == "a 4\nb 2\n"
        assert not (directory / f"{name}_lengths_shape").exists()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("metric_type", ["int", "float", "numerical", "str"])
def test_metric_types_through_dataset_and_collation(tmp_path, streaming, metric_type):
    (tmp_path / "metrics.scp").write_text('a {"mos": "3"}\nb {}\n')
    (tmp_path / "audio.scp").write_text("a 4\nb 6\n")
    cls = IterableESPnetDataset if streaming else ESPnetDataset
    dataset = cls(
        [
            (str(tmp_path / "metrics.scp"), "metrics", "metric"),
            (str(tmp_path / "audio.scp"), "audio", "rand_float"),
        ],
        preprocess=UniversaProcessor(train=False, metric2type={"mos": metric_type}),
        float_dtype="float64",
    )

    def samples():
        return list(dataset) if streaming else [dataset["a"], dataset["b"]]

    if metric_type == "str":
        with pytest.raises(ValueError, match="String metric.*tokenizing preprocessor"):
            samples()
    else:
        _, batch = UniversaCollateFn(["mos"], metric_pad_value=-100)(samples())
        assert batch["audio"].dtype == torch.float64
        assert batch["metrics"]["mos"].dtype == torch.float32
        torch.testing.assert_close(batch["metrics"]["mos"], torch.tensor([3.0, -100.0]))


@pytest.mark.parametrize("preprocess", [None, lambda uid, data: data])
def test_streaming_rejects_unresolved_missing_audio(tmp_path, preprocess):
    scp = tmp_path / "wav.scp"
    scp.write_text("a None\n")
    dataset = IterableESPnetDataset(
        [(str(scp), "audio", "sound")], preprocess=preprocess
    )
    with pytest.raises(RuntimeError, match='Missing value for "audio"'):
        next(iter(dataset))


@pytest.mark.parametrize("force_single_channel", [False, True])
@pytest.mark.parametrize("channels", [None, 1, 2])
def test_audio_rank_consistent_across_training_and_evaluation(
    channels, force_single_channel
):
    shape = (16,) if channels is None else (16, channels)
    audio = np.full(shape, 2.0, dtype=np.float32)
    outputs = [
        UniversaProcessor(
            train=train,
            force_single_channel=force_single_channel,
            audio_volume_normalize=0.5,
        )("a", {"audio": audio.copy()})
        for train in (False, True)
    ]
    expected = (16,) if force_single_channel else shape
    for result in outputs:
        assert result["audio"].shape == expected
        np.testing.assert_allclose(result["audio"], 0.5)


def test_missing_required_audio_rejected():
    with pytest.raises(ValueError, match="Missing required audio"):
        UniversaProcessor(train=False)("a", {"audio": None})


def test_text_preprocessing_and_numeric_labels():
    processor = UniversaProcessor(
        train=False,
        token_type="char",
        token_list=["<unk>", "a", "b"],
        metric2type={"mos": "float"},
    )
    result = processor("a", {"ref_text": "ab", "metrics": {"mos": "2.5"}})
    np.testing.assert_array_equal(result["ref_text"], [1, 2])
    assert result["metrics"] == {"mos": 2.5}
    np.testing.assert_array_equal(processor("a", result)["ref_text"], [1, 2])
    np.testing.assert_array_equal(processor("a", {"ref_text": None})["ref_text"], [0])


@pytest.mark.parametrize("streaming", [False, True])
def test_generic_json_annotations(tmp_path, streaming):
    from espnet2.fileio.json_scp import JsonScpReader
    from espnet2.fileio.metric_scp import MetricReader

    scp = tmp_path / "annotations.scp"
    scp.write_text('a {"confidence": 0.75, "count": 3}\nb {"count": 2}\n')
    (tmp_path / "audio.scp").write_text("a 4\nb 6\n")
    reader = JsonScpReader(scp)
    assert MetricReader is JsonScpReader
    assert len(reader) == 2 and "a" in reader
    assert list(reader.keys()) == list(reader) == ["a", "b"]
    cls = IterableESPnetDataset if streaming else ESPnetDataset
    dataset = cls(
        [
            (str(scp), "annotations", "json"),
            (str(tmp_path / "audio.scp"), "audio", "rand_float"),
        ]
    )
    data = list(dataset) if streaming else [dataset["a"], dataset["b"]]
    _, batch = MappingCollateFn(
        {"annotations": ["confidence", "count"]}, mapping_pad_value=-100
    )(data)
    torch.testing.assert_close(
        batch["annotations"]["confidence"], torch.tensor([0.75, -100.0])
    )
    torch.testing.assert_close(batch["annotations"]["count"], torch.tensor([3.0, 2.0]))
    assert batch["audio"].shape == (2, 6)
    assert batch["audio_lengths"].tolist() == [4, 6]


@pytest.mark.parametrize("value", ["[1, 2]", "null", '"text"'])
def test_json_reader_rejects_non_object(tmp_path, value):
    from espnet2.fileio.json_scp import JsonScpReader

    scp = tmp_path / "annotations.scp"
    scp.write_text(f"a {value}\n")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        JsonScpReader(scp)["a"]


def test_mapping_collator_rejects_unencoded_categories():
    with pytest.raises(ValueError, match="annotations.language.*numeric scalar"):
        MappingCollateFn({"annotations": ["language"]})(
            [("a", {"audio": np.zeros(4), "annotations": {"language": "eng"}})]
        )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("name", ["audio", "ref_audio", "other_audio"])
def test_missing_audio_is_optional_only_for_configured_reference(
    tmp_path, streaming, name
):
    scp = tmp_path / "wav.scp"
    scp.write_text("a None\n")
    cls = IterableESPnetDataset if streaming else ESPnetDataset
    dataset = cls(
        [(str(scp), name, "sound")], preprocess=UniversaProcessor(train=False)
    )

    def read():
        return next(iter(dataset)) if streaming else dataset["a"]

    if name == "ref_audio":
        assert np.count_nonzero(read()[1][name]) == 0
    else:
        with pytest.raises((ValueError, RuntimeError), match=f"Missing.*{name}"):
            read()


def test_streaming_none_text_is_not_missing_audio(tmp_path):
    scp = tmp_path / "text"
    scp.write_text("a None\n")

    def preprocess(uid, data):
        assert data["text"] == "None"
        return {"text": np.array([1], dtype=np.int64)}

    dataset = IterableESPnetDataset([(str(scp), "text", "text")], preprocess=preprocess)
    assert next(iter(dataset))[1]["text"].tolist() == [1]
