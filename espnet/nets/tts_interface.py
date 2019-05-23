class TTSInterface(object):
    """TTS Interface for ESPnet model implementation"""

    @staticmethod
    def add_arguments(parser):
        return parser

    def forward(self, xs, ilens, ys, olens, spembs=None):
        """TTS forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens:
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attention weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        """
        raise NotImplementedError("forward method is not implemented")

    def inference(self, x, inference_args, spemb=None):
        """Generates the sequence of features given the sequences of characters

        :param torch.Tensor x: the sequence of characters (T)
        :param Namespace inference_args: argments containing following attributes
            (float) threshold: threshold in inference
            (float) minlenratio: minimum length ratio in inference
            (float) maxlenratio: maximum length ratio in inference
        :param torch.Tensor spemb: speaker embedding vector (spk_embed_dim)
        :return: the sequence of features (L, odim)
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        :return: the sequence of attention weight (L, T)
        :rtype: torch.Tensor
        """
        raise NotImplementedError("inference method is not implemented")

    def calculate_all_attentions(self, xs, ilens, ys, spembs=None):
        """TTS attention caluculation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: attention weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    @property
    def attention_plot_class(self):
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport
