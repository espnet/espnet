from abc import ABC

import torch

from espnet.nets.pytorch_backend.rnn.attentions import NoAtt, AttDot, AttAdd, \
    AttLoc, AttCov, AttLoc2D, AttLocRec, AttCovLoc, AttMultiHeadDot, \
    AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc, AttForward, \
    AttForwardTA
from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention


class AbsAttention(torch.nn.Module, ABC):
    # A marker class to represent "Attention" object
    pass


# TODO(kamo): Using tricky way such as register() to keep espnet/ as it is.
#  Each class should inherit the abs class originally.
# See also: calculate_all_attentions()
AbsAttention.register(MultiHeadedAttention)
AbsAttention.register(NoAtt)
AbsAttention.register(AttDot)
AbsAttention.register(AttAdd)
AbsAttention.register(AttLoc)
AbsAttention.register(AttCov)
AbsAttention.register(AttLoc2D)
AbsAttention.register(AttLocRec)
AbsAttention.register(AttCovLoc)
AbsAttention.register(AttMultiHeadDot)
AbsAttention.register(AttMultiHeadAdd)
AbsAttention.register(AttMultiHeadLoc)
AbsAttention.register(AttMultiHeadMultiResLoc)
AbsAttention.register(AttForward)
AbsAttention.register(AttForwardTA)
