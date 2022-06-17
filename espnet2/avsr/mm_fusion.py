import torch
from typeguard import check_argument_types


def concat_fusion(
    speech_feats: torch.Tensor, vision_feats: torch.Tensor, feat_lengths: torch.Tensor
):
    """
    Concatenates vision features and speech features
    """
    B, T, F1 = speech_feats.size()
    _, _, F2 = vision_feats.size()
    assert (B, T, F2) == vision_feats.size(), vision_feats.shape
    concat_feats = torch.cat([speech_feats, vision_feats], dim=-1)
    assert concat_feats.size() == (B, T, F1 + F2), concat_feats.shape
    return concat_feats, feat_lengths


FUSION_OPTIONS = {
    "concat": concat_fusion,
}


class MM_Fuser:
    def __init__(self, fusion_type):
        if fusion_type not in FUSION_OPTIONS.keys():
            raise ValueError("fusion_type should be one of {}".format(FUSION_OPTIONS))
        self.fusion_func = FUSION_OPTIONS[fusion_type]

    def fuse(
        self, speech: torch.Tensor, vision: torch.Tensor, feat_lengths: torch.Tensor
    ):
        """
        @requires: speech and vision features must be aligned
        """
        assert check_argument_types()
        assert speech.size(0) == vision.size(0), (speech.shape, vision.shape)
        assert speech.size(1) == vision.size(1), (speech.shape, vision.shape)
        assert feat_lengths.dim() == 1, feat_lengths.shape
        return self.fusion_func(speech, vision, feat_lengths)
