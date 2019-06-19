import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """
    def __init__(self, idim, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class PositionwiseConv1d(torch.nn.Module):
    """Positionwise conv1d

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        (https://arxiv.org/pdf/1905.09263.pdf)

    :param int in_chans: number of input channels
    :param int hidden_chans: number of hidden channels
    :param int kernel_size: kernel size of conv1d
    :param float dropout_rate: dropout rate
    """
    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(PositionwiseConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)
