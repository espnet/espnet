class MTInterface(object):
    """MT Interface for ESPnet model implementation"""

    @staticmethod
    def add_arguments(parser):
        return parser

    def forward(self, xs, ilens, ys):
        '''compute loss for training

        :param xs:
            For pytorch, batch of padded source sequences torch.Tensor (B, Tmax, idim)
            For chainer, list of source sequences chainer.Variable
        :param ilens: batch of lengths of source sequences (B)
            For pytorch, torch.Tensor
            For chainer, list of int
        :param ys:
            For pytorch, batch of padded source sequences torch.Tensor (B, Lmax)
            For chainer, list of source sequences chainer.Variable
        :return: loss value
        :rtype: torch.Tensor for pytorch, chainer.Variable for chainer
        '''
        raise NotImplementedError("forward method is not implemented")

    def translate(self, x, trans_args, char_list=None, rnnlm=None):
        '''translate x for evaluation

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace trans_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        raise NotImplementedError("translate method is not implemented")

    def calculate_all_attentions(self, xs, ilens, ys):
        '''attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    @property
    def attention_plot_class(self):
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport
