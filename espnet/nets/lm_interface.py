"""Language model interface."""

from espnet.nets.scorer_interface import ScorerInterface
from espnet.utils.dynamic_import import dynamic_import


class LMInterface(ScorerInterface):
    """LM Interface for ESPnet model implementation."""

    @staticmethod
    def add_arguments(parser):
        """Add arguments to command line argument parser."""
        return parser

    def forward(self, x, t):
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)
            t (torch.Tensor): Target ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        raise NotImplementedError("forward method is not implemented")


predefined_lms = {
    "pytorch": {
        "default": "espnet.nets.pytorch_backend.lm.default:DefaultRNNLM",
        "seq_rnn": "espnet.nets.pytorch_backend.lm.seq_rnn:SequentialRNNLM",
        "transformer": "espnet.nets.pytorch_backend.lm.transformer:TransformerLM",
    },
    "chainer": {
        "default": "espnet.lm.chainer_backend.lm:DefaultRNNLM"
    }
}


def dynamic_import_lm(module, backend):
    """Import LM class dynamically.

    Args:
        module (str): module_name:class_name or alias in `predefined_lms`
        backend (str): NN backend. e.g., pytorch, chainer

    Returns:
        type: LM class

    """
    model_class = dynamic_import(module, predefined_lms.get(backend, dict()))
    assert issubclass(model_class, LMInterface), f"{module} does not implement LMInterface"
    return model_class
