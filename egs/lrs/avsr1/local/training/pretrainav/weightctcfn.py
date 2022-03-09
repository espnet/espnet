import torch


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization

        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class Net(torch.nn.Module):
    def __init__(self, adim):
        super(Net, self).__init__()
        self.weightlayer1 = torch.nn.Linear(adim, int(adim / 2))
        self.norm1 = LayerNorm(int(adim / 2))
        self.weightlayer2 = torch.nn.Linear(int(adim / 2), int(adim / 4))
        self.norm2 = LayerNorm(int(adim / 4))
        self.weightlayer3 = torch.nn.Linear(int(adim / 4), int(adim / 8))
        self.norm3 = LayerNorm(int(adim / 8))
        self.weightlayer4 = torch.nn.Linear(int(adim / 8), 2)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        input = self.norm1(self.relu(self.weightlayer1(input)))
        input = self.norm2(self.relu(self.weightlayer2(input)))
        input = self.norm3(self.relu(self.weightlayer3(input)))
        output = self.weightlayer4(input)

        return output
