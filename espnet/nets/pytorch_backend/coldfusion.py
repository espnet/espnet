# Inspired by https://github.com/pytorch/translate/blob/master/pytorch_translate/multi_model.py

import torch

import torch.nn.functional as F


class ColdFusionLayer(torch.nn.Module):
    """A Layer for Cold Fusion"""

    def __init__(self, decoder_size, vocab_size, hidden_size=256):
        """
        Inits the layer
        :param int decoder_size: The decoder output size
        :param int vocab_size: The vocabulary (output) size
        :param int hidden_size: The hidden size of this layer
        """
        super(ColdFusionLayer, self).__init__()
        self.hidden_layer = torch.nn.Linear(in_features=vocab_size, out_features=hidden_size, bias=False)
        self.gating_layer = torch.nn.Linear(in_features=hidden_size + decoder_size, out_features=hidden_size, bias=True)
        self.output_layer = torch.nn.Linear(in_features=decoder_size + hidden_size, out_features=vocab_size, bias=True)

    def forward(self, decoder_output, lm_output):
        """Feedforward

        :param torch.Tensor decoder_output: The output of the decoder
        :param torch.Tensor lm_output: The output of the language model
        :return: torch.Tensor The output of this layer
        """
        lm_max, _ = torch.max(lm_output, dim=1, keepdim=True)
        lm_output = lm_output - lm_max
        h_lm = self.hidden_layer(lm_output)
        h_lm = F.relu(h_lm)
        g = self.gating_layer(torch.cat([decoder_output, h_lm], dim=1))
        g = F.sigmoid(g)
        s_cf = torch.cat([decoder_output, g * h_lm], dim=1)
        r_cf = self.output_layer(s_cf)
        r_cf = F.relu(r_cf)
        return r_cf
