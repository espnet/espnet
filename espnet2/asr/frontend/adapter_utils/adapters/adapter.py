import torch
import torch.nn as nn

try:
    from fairseq import utils
    from fairseq.modules import LayerNorm

    is_fairseq_available = True
except ImportError:
    is_fairseq_available = False




class Adapter(nn.Module):
    def __init__(
        self,
        orig_dim: int,
        down_dim: int,
        layer_norm: str = None,
        activation_fn: str = "gelu",
    ) -> None:
        """
        * orig_dim - original dimension.
        * down_dim - dimension of Adapter's intermediate down-projection.
        * layer_norm - location of LayerNorm layer ("first" or "last", default=None). If `None`, LayerNorm is not used.
        * activation_fn - activation type (default="gelu").
        """
        super().__init__()
        if not is_fairseq_available:
            raise ImportError(
                "`fairseq` is not available. Please install it via `pip install"
                " fairseq` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_fairseq.sh`."
            )
        
        self.down_projection = nn.Linear(orig_dim, down_dim)
        self.up_projection = nn.Linear(down_dim, orig_dim)
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.xavier_uniform_(self.up_projection.weight)
        self.activation = utils.get_activation_fn(activation_fn)

        self.layer_norm_opt = layer_norm
        if layer_norm is not None:
            self.layer_norm = LayerNorm(orig_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm_opt == "first":
            x = self.layer_norm(x)

        x = self.down_projection(x)
        x = self.activation(x)
        x = self.up_projection(x)

        if self.layer_norm_opt == "last":
            x = self.layer_norm(x)

        return x
