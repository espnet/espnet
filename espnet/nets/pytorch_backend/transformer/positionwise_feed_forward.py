import torch


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class PositionwiseConv1d(torch.nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dropout):
        super(PositionwiseConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(d_model, d_ff, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(d_ff, d_model, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)
