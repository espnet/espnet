from abc import ABC

import torch

from espnet.nets.pytorch_backend.rnn.attentions import AttAdd
from espnet.nets.pytorch_backend.rnn.attentions import AttCov
from espnet.nets.pytorch_backend.rnn.attentions import AttCovLoc
from espnet.nets.pytorch_backend.rnn.attentions import AttDot
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc2D
from espnet.nets.pytorch_backend.rnn.attentions import AttLocRec
from espnet.nets.pytorch_backend.rnn.attentions import AttMultiHeadAdd
from espnet.nets.pytorch_backend.rnn.attentions import AttMultiHeadDot
from espnet.nets.pytorch_backend.rnn.attentions import AttMultiHeadLoc
from espnet.nets.pytorch_backend.rnn.attentions import AttMultiHeadMultiResLoc
from espnet.nets.pytorch_backend.rnn.attentions import NoAtt
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class AbsAttention(torch.nn.Module, ABC):
    """A marker class to represent "Attention" object

    See also: calculate_all_attentions()
    """


# TODO(kamo): Using tricky way such as register() to keep espnet/ as it is.
#  Each class should inherit the abs class originally.
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
