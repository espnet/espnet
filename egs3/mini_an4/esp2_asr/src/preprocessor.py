from __future__ import annotations

from pathlib import Path
from typing import Collection, Dict, Iterable, Optional, Sequence, Union

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor, CommonPreprocessor


class MiniAn4TokenizeSpeedPerturbPreprocessor(AbsPreprocessor):
    """Custom preprocessor for mini_an4 integration tests."""

    def __init__(
        self,
        train: bool,
        token_type: Optional[str] = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        fs: int = 0,
        speech_name: str = "speech",
        text_name: str = "text",
        speed_perturb_factors: Optional[Sequence[float]] = None,
        speed_perturb_prob: float = 0.0,
    ):
        super().__init__(train=train)
        self._delegate = CommonPreprocessor(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            fs=fs,
            speech_name=speech_name,
            text_name=text_name,
            data_aug_effects=self._build_speed_perturb_effects(
                train=train,
                speed_perturb_factors=speed_perturb_factors,
                speed_perturb_prob=speed_perturb_prob,
            ),
            data_aug_num=[1, 1],
            data_aug_prob=1.0 if train and speed_perturb_prob > 0.0 else 0.0,
        )

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Apply recipe preprocessing to one sample."""
        return self._delegate(uid, data)

    def _build_speed_perturb_effects(
        self,
        train: bool,
        speed_perturb_factors: Optional[Sequence[float]],
        speed_perturb_prob: float,
    ):
        """Build a train-only speed-perturb effect group."""
        if not train or speed_perturb_prob <= 0.0 or not speed_perturb_factors:
            return None

        effects = []
        for factor in speed_perturb_factors:
            effects.append([1.0, "speed_perturb", {"factor": factor}])
        return [[speed_perturb_prob, effects]]
