import torch


class MultiSequential(torch.nn.Sequential):
    def forward(self, *args):
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """repeat module N times

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])
