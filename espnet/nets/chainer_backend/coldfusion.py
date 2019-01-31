import chainer
import chainer.functions as F
import chainer.links as L


class ColdFusionLayer(chainer.Chain):
    """A layer for ColdFusion"""

    def __init__(self, decoder_size, vocab_size, hidden_size=256):
        """Init the ColdFusion layer

        :param decoder_size: The number of decoder hidden units
        :param vocab_size: The output vocabulary size
        :param hidden_size: The hidden size of this layer
        """
        super(ColdFusionLayer, self).__init__()
        self.decoder_size = decoder_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        with self.init_scope():
            self.hidden_layer = L.Linear(vocab_size, hidden_size, nobias=True)
            self.gating_layer = L.Linear(hidden_size + decoder_size, hidden_size, nobias=False)
            self.output_layer = L.Linear(hidden_size + decoder_size, vocab_size, nobias=False)

    def __call__(self, decoder_output, lm_output):
        """Feedforward

        :param decoder_output: The output of the decoder
        :param lm_output: The output of the language model
        :return: The output of this layer
        """
        lm_max = F.max(lm_output, axis=1, keepdims=True)
        lm_output = lm_output - lm_max
        h_lm = F.relu(self.hidden_layer(lm_output))
        g = F.sigmoid(self.gating_layer(F.concat([decoder_output, h_lm], axis=1)))
        s_cf = F.concat([decoder_output, g * h_lm], axis=1)
        r_cf = F.relu(self.output_layer(s_cf))
        return r_cf
