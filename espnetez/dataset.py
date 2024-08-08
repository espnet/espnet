from typing import Dict, Tuple, Union

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    def __init__(self, dataset, data_info):
        self.dataset = dataset
        self.data_info = data_info

    def has_name(self, name) -> bool:
        return name in self.data_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.data_info.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict]:
        idx = int(uid)
        return (
            str(uid),
            {k: v(self.dataset[idx]) for k, v in self.data_info.items()},
        )

    def __len__(self) -> int:
        return len(self.dataset)


class ESPnetEZLhotseDataset(ESPnetEZDataset):
    """
    A simple wrapper class for supporting lhotse manifests in ESPNet-EZ.

    Args:
        supervisions: lhotse.SupervisionSet, path to the lhotse supervision manifest.
        recordings: lhotse.RecordingSet, path to the corresponding lhotse recordings manifest.
    """

    def __init__(self, supervisions, recordings):

        # lazy importing for lhotse
        try:
            import lhotse
        except ImportError:
            raise ImportError(
                "Cannot import Lhotse. Have you installed it correctly ? "
                "See https://github.com/lhotse-speech/lhotse."
            )

        supervisions = lhotse.load_manifest(supervisions)
        assert isinstance(supervisions, lhotse.SupervisionSet)
        recordings = lhotse.load_manifest(recordings)
        assert isinstance(recordings, lhotse.RecordingSet)

        # these have to be mapped to dataset and data_info now.
        # TODO

        # call the parent init
        ESPnetEZDataset.__init__(dataset, data_info)
