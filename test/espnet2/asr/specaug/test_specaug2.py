import pytest
import torch

from espnet2.asr.specaug.specaug2 import SpecAug2


@pytest.mark.parametrize("apply_time_warp", [False, True])
@pytest.mark.parametrize("apply_freq_mask", [False, True])
@pytest.mark.parametrize("apply_time_mask", [False, True])
def test_SpecAuc2(apply_time_warp, apply_freq_mask, apply_time_mask):
    if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
        with pytest.raises(ValueError):
            specaug = SpecAug2(
                apply_time_warp=apply_time_warp,
                apply_freq_mask=apply_freq_mask,
                apply_time_mask=apply_time_mask,
            )
    else:
        specaug = SpecAug2(
            apply_time_warp=apply_time_warp,
            apply_freq_mask=apply_freq_mask,
            apply_time_mask=apply_time_mask,
        )
        x = torch.randn(2, 1000, 80)
        specaug(x)


@pytest.mark.parametrize("apply_time_warp", [False, True])
@pytest.mark.parametrize("apply_freq_mask", [False, True])
@pytest.mark.parametrize("apply_time_mask", [False, True])
def test_SpecAuc2_repr(apply_time_warp, apply_freq_mask, apply_time_mask):
    if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
        return
    specaug = SpecAug2(
        apply_time_warp=apply_time_warp,
        apply_freq_mask=apply_freq_mask,
        apply_time_mask=apply_time_mask,
    )
    print(specaug)
