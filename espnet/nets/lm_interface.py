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
