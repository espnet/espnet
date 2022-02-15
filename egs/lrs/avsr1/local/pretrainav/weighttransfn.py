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
        
class transformerNet(torch.nn.Module):
    def __init__(self, adim, odim):
        super(transformerNet, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.layer0 = torch.nn.Linear(adim, 2048)
        self.norm0 = LayerNorm(2048)        
        self.layer1 = torch.nn.Linear(2048, 1024)
        self.norm1 = LayerNorm(1024)
        self.layer2 = torch.nn.Linear(1024, 512)
        self.norm2 = LayerNorm(512)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(512, odim)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input):
        input = self.norm0(self.dropout(torch.relu(self.layer0(input))))    
        input = self.norm1(self.dropout(torch.relu(self.layer1(input))))
        input = self.norm2(self.dropout(torch.tanh(self.layer2(input))))
        input = self.fc(input)
        output = self.softmax(input)

        return output
        
        
        
######blstm
class ctcNet(torch.nn.Module):
    def __init__(self, adim, odim):
        super(ctcNet, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.layer1 = torch.nn.Linear(adim, 1024)
        self.norm1 = LayerNorm(1024)
        self.layer2 = torch.nn.Linear(1024, 512)
        self.norm2 = LayerNorm(512)
        self.layer4 = torch.nn.LSTM(input_size=512,
                              hidden_size=512,
                              num_layers=3,
                              batch_first=True,
                              bidirectional=True)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(512 * 2, odim)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input):
        input = self.norm1(self.dropout(torch.relu(self.layer1(input))))
        input = self.norm2(self.dropout(torch.relu(self.layer2(input))))
        input, _ = self.layer4(input)
        input = self.fc(self.dropout(torch.tanh(input)))
        output = self.softmax(input)

        return output



