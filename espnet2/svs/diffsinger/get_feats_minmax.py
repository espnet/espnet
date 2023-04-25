from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types


class GetFeatsMinMax(torch.nn.Module):
    """Get Min and Max of Feats

    Args:
        stats_file: npy file
    """

    def __init__(
        self,
        stats_file: Union[Path, str],
    ):
        assert check_argument_types()
        super().__init__()
        stats_file = Path(stats_file)

        self.stats_file = stats_file
        stats = np.load(stats_file)
        if isinstance(stats, np.ndarray):
            # Kaldi like stats
            raise TypeError("Not support Kaldi style type")
        else:
            # New style: Npz file
            feats_min = stats["feats_min"]
            feats_max = stats["feats_max"]

        if isinstance(feats_min, np.ndarray):
            feats_min = torch.from_numpy(feats_min)
        if isinstance(feats_max, np.ndarray):
            feats_max = torch.from_numpy(feats_max)

        self.register_buffer("feats_min", feats_min)
        self.register_buffer("feats_max", feats_max)

    def extra_repr(self):
        return (
            f"stats_file={self.stats_file}, "
            f"feats_min={self.feats_min}, feats_max={self.feats_max}"
        )

    def forward(
        self,
    ) -> Dict[str, torch.Tensor]:
        return {
            "feats_min": self.feats_min,
            "feats_max": self.feats_max,
        }
