import numpy as np
import pytest
import torch

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.main_funcs.collect_stats import _utterance, collect_stats
from espnet2.train.abs_espnet_model import AbsESPnetModel


class FrontendModel(AbsESPnetModel):
    """Model whose collect_feats runs an STFT/mel frontend.

    In a padded batch, the last frames of a short utterance see the padding.
    """

    def __init__(self):
        super().__init__()
        self.frontend = DefaultFrontend(fs=16000, n_fft=64, hop_length=16, n_mels=8)

    def forward(self, speech, speech_lengths, **kwargs):
        raise NotImplementedError

    def collect_feats(self, speech, speech_lengths, **kwargs):
        feats, feats_lengths = self.frontend(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}


def _batches(speech, groups):
    """(keys, padded batch) for each group of keys."""
    out = []
    for keys in groups:
        lengths = torch.tensor([len(speech[k]) for k in keys])
        padded = torch.zeros(len(keys), int(lengths.max()))
        for i, k in enumerate(keys):
            padded[i, : lengths[i]] = speech[k]
        out.append((keys, {"speech": padded, "speech_lengths": lengths}))
    return out


def _stats(tmp_path, name, speech, groups):
    out = tmp_path / name
    batches = _batches(speech, groups)
    collect_stats(FrontendModel().eval(), batches, batches, out, 0, 10, True)
    return out / "train"


@pytest.mark.execution_timeout(20)
def test_stats_do_not_depend_on_batches(tmp_path):
    torch.manual_seed(0)
    lengths = {"a": 1000, "b": 400, "c": 777, "d": 250, "e": 1300}
    speech = {k: torch.randn(n) for k, n in lengths.items()}
    groupings = {
        "one_per_batch": [["a"], ["b"], ["c"], ["d"], ["e"]],
        "pairs": [["a", "b"], ["c", "d"], ["e"]],
        "other_pairs": [["a", "e"], ["b", "d"], ["c"]],
        "all": [["a", "b", "c", "d", "e"]],
    }
    dirs = {n: _stats(tmp_path, n, speech, g) for n, g in groupings.items()}

    ref = np.load(dirs["one_per_batch"] / "feats_stats.npz")
    assert ref["sum"].dtype == np.float64
    for name, d in dirs.items():
        stats = np.load(d / "feats_stats.npz")
        assert stats["count"] == ref["count"]
        np.testing.assert_allclose(stats["sum"], ref["sum"], rtol=1e-12)
        np.testing.assert_allclose(stats["sum_square"], ref["sum_square"], rtol=1e-12)
        # Lines are in batch order; aggregate_stats_dirs sorts them
        shapes = sorted((d / "speech_shape").read_text().splitlines())
        ref_shapes = (dirs["one_per_batch"] / "speech_shape").read_text()
        assert shapes == sorted(ref_shapes.splitlines())

    # The written features are those of each utterance on its own
    for name, d in dirs.items():
        scp = dict(line.split() for line in (d / "collect_feats/feats.scp").open())
        for k in lengths:
            alone, _ = (
                FrontendModel()
                .eval()
                .frontend(speech[k][None], torch.tensor([lengths[k]]))
            )
            np.testing.assert_array_equal(np.load(scp[k]), alone[0].numpy(), name)


def test_utterance_cuts_padding_and_keeps_other_values():
    batch = {
        "speech": torch.arange(8.0).view(2, 4),
        "speech_lengths": torch.tensor([4, 2]),
        "spk": torch.tensor([[7], [9]]),
        "metrics": {"mos": torch.tensor([1.0, 2.0])},
        "scalar": torch.tensor(3.0),
    }
    one = _utterance(batch, 1, 2)
    assert one["speech"].tolist() == [[4.0, 5.0]]
    assert one["speech_lengths"].tolist() == [2]
    assert one["spk"].tolist() == [[9]]
    assert one["metrics"] is batch["metrics"]
    assert one["scalar"] is batch["scalar"]
