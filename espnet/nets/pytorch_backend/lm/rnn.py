import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.beam_search import DecoderInterface


class FastRNNLM(LMInterface, DecoderInterface, torch.nn.Module):
    """Legacy RNNLM wrapper to compute reduce framewise loss values

    Args:
        n_vocab (int): The size of the vocabulary
        args (argparse.Namespace): configurations. see `add_arguments`

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--type', type=str, default="lstm", nargs='?', choices=['lstm', 'gru'],
                            help="Which type of RNN to use")
        parser.add_argument('--layer', '-l', type=int, default=2,
                            help='Number of hidden layers')
        parser.add_argument('--unit', '-u', type=int, default=650,
                            help='Number of hidden units')
        parser.add_argument('--dropout-rate', type=float, default=0.5,
                            help='dropout probability')
        return parser

    def __init__(self, n_vocab, args):
        torch.nn.Module.__init__(self)
        self.setup(
            rnn_type=args.type.upper(),
            ntoken=n_vocab,
            ninp=args.unit,
            nhid=args.unit,
            nlayers=args.layer,
            dropout=args.dropout_rate)

    def setup(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        # NOTE: original init in pytorch/examples
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # NOTE: our legacy.py:RNNLM init
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def forward(self, x, t):
        mask = (x != 0)
        y = self.before_loss(x, None)[0]
        y = y * mask.float().unsqueeze(-1)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1))
        count = mask.sum().int()
        return loss, count

    def before_loss(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_state(self, x):
        bsz = 1
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def score(self, y, state, x):
        y, new_state = self.before_loss(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state
