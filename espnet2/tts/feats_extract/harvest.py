# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""F0 extractor using harvest algorithm."""

import numpy as np
import pyworld
import torch

from espnet2.tts.feats_extract.dio import Dio


class Harvest(Dio):
    """F0 estimation with harvest algorithm.

    This is f0 extractor based on `Harvest`_ algorithm introduced in `WORLD:
    a vocoder-based high-quality speech synthesis system for real-time applications`_.

    .. _`WORLD: a vocoder-based high-quality speech synthesis system for real-time
        applications`: https://doi.org/10.1587/transinf.2015EDP7457

    .. _`Harvest`: https://doi.org/10.21437/Interspeech.2017-68

    Note:
        This module is based on NumPy implementation. Therefore, the computational graph
        is not connected.

    Todo:
        Replace this module with PyTorch-based implementation.

    """

    def _calculate_f0(self, input: torch.Tensor) -> torch.Tensor:
        x = input.cpu().numpy().astype(np.double)
        f0, timeaxis = pyworld.harvest(
            x,
            self.fs,
            f0_floor=self.f0min,
            f0_ceil=self.f0max,
            frame_period=self.frame_period,
        )
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return input.new_tensor(f0.reshape(-1), dtype=torch.float)
