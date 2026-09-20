import numpy as np
import pytest
import soundfile
import torch

from espnet2.train.collate_fn import UniversaCollateFn
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
