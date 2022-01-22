import pytest
import torch

from espnet2.asr.specaug.specaug import SpecAug


@pytest.mark.parametrize("apply_time_warp", [False, True])
@pytest.mark.parametrize("apply_freq_mask", [False, True])
@pytest.mark.parametrize("apply_time_mask", [False, True])
@pytest.mark.parametrize("time_mask_width_range", [None, 100, (0, 100)])
@pytest.mark.parametrize("time_mask_width_ratio_range", [None, 0.1, (0.0, 0.1)])
def test_SpecAuc(
    apply_time_warp,
    apply_freq_mask,
    apply_time_mask,
    time_mask_width_range,
    time_mask_width_ratio_range,
):
    if (
        (not apply_time_warp and not apply_time_mask and not apply_freq_mask)
        or (
            apply_time_mask
            and time_mask_width_range is None
            and time_mask_width_ratio_range is None
        )
        or (
            apply_time_mask
            and time_mask_width_range is not None
            and time_mask_width_ratio_range is not None
        )
    ):
        with pytest.raises(ValueError):
            specaug = SpecAug(
                apply_time_warp=apply_time_warp,
                apply_freq_mask=apply_freq_mask,
                apply_time_mask=apply_time_mask,
                time_mask_width_range=time_mask_width_range,
                time_mask_width_ratio_range=time_mask_width_ratio_range,
            )
    else:
        specaug = SpecAug(
            apply_time_warp=apply_time_warp,
            apply_freq_mask=apply_freq_mask,
            apply_time_mask=apply_time_mask,
            time_mask_width_range=time_mask_width_range,
            time_mask_width_ratio_range=time_mask_width_ratio_range,
        )
        x = torch.randn(2, 1000, 80)
        specaug(x)


@pytest.mark.parametrize("apply_time_warp", [False, True])
@pytest.mark.parametrize("apply_freq_mask", [False, True])
@pytest.mark.parametrize("apply_time_mask", [False, True])
@pytest.mark.parametrize("time_mask_width_range", [None, 100, (0, 100)])
@pytest.mark.parametrize("time_mask_width_ratio_range", [None, 0.1, (0.0, 0.1)])
def test_SpecAuc_repr(
    apply_time_warp,
    apply_freq_mask,
    apply_time_mask,
    time_mask_width_range,
    time_mask_width_ratio_range,
):
    if (
        (not apply_time_warp and not apply_time_mask and not apply_freq_mask)
        or (
            apply_time_mask
            and time_mask_width_range is None
            and time_mask_width_ratio_range is None
        )
        or (
            apply_time_mask
            and time_mask_width_range is not None
            and time_mask_width_ratio_range is not None
        )
    ):
        return
    specaug = SpecAug(
        apply_time_warp=apply_time_warp,
        apply_freq_mask=apply_freq_mask,
        apply_time_mask=apply_time_mask,
        time_mask_width_range=time_mask_width_range,
        time_mask_width_ratio_range=time_mask_width_ratio_range,
    )
    print(specaug)
