from espnet.utils.dynamic_import import dynamic_import


class LMInterface:
    """LM Interface for ESPnet model implementation"""

    @staticmethod
    def add_arguments(parser):
        return parser

    def forward(self, x, t):
        """Compute LM training loss value for sequences

        Args:
            x (torch.Tensor): Input ids of torch.int64. (batch, len)
            t (torch.Tensor): Target ids of torch.int64. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of
                the reduced loss value along time (scalar)
                and the number of valid loss values (scalar)
        """
        raise NotImplementedError("forward method is not implemented")

    def init_state(self):
        pass

    def select_state(self, ids):
        pass

    def score(self, x, state):
        return 0


predefined_lms = {
    "pytorch": {
        "legacy": "espnet.nets.pytorch_backend.lm.legacy:LegacyRNNLM"
    },
    "chainer": {
        "legacy": "espnet.nets.pytorch_backend.lm.legacy:LegacyRNNLM"
    }
}


def dynamic_import_lm(module, backend=None):
    """Dynamic import LM

    Args:
        module (str): module_name:class_name or alias in `predefined_lms`
        backend (str): NN backend. e.g., pytorch, chainer

    Returns:
        type: LM class
    """
    model_class = dynamic_import(module, predefined_lms.get(backend, dict()))
    assert issubclass(model_class, LMInterface), f"{module} does not implement LMInterface"
    return model_class
