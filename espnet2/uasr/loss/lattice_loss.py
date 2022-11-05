import logging
import torch

from typeguard import check_argument_types

from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.uasr.loss.abs_loss import AbsUASRLoss

try:
    import k2
    from icefall.decode import get_lattice
except ImportError or ModuleNotFoundError:
    k2 = None
    get_lattice = None


class UASRLatticeLoss(AbsUASRLoss):
    """gradient penalty for UASR."""

    def __init__(
        self,
        HLG: str,
        weight: float = 1.0,
        use_log_semiring: str2bool = True,
    ):
        super().__init__()
        assert check_argument_types()
        assert (
            k2 is not None and get_lattice is not None
        ), "k2/icefall is not correctly installed, use 'tools/installers/install_k2.sh' to install k2"

        device = torch.device("cuda", 0)
        decoding_graph = k2.Fsa.from_dict(torch.load(HLG))
        self.decoding_graph = decoding_graph.to(device)
        self.weight = weight
        self.use_log_semiring = use_log_semiring

        self.search_beam = 15
        self.output_beam = 7
        self.min_active_states = 200
        self.max_active_states = 7000

    def forward(
        self,
        generated_sample: torch.Tensor,
        generated_sample_padding_maks: torch.Tensor,
        is_training: str2bool,
        is_discrimininative_step: str2bool,
    ):
        """Forward.

        Args:
        """
        if self.weight > 0 and not is_discrimininative_step and is_training:
            nnet_output = torch.log(generated_sample)
            batch_size, time_length, channel = nnet_output.shape

            supervision_segments = torch.zeros([batch_size, 3])
            for i in range(batch_size):
                num_frames = time_length - generated_sample_padding_maks[i].sum()
                supervision_segments[i][0] = i
                supervision_segments[i][2] = num_frames
            supervision_segments = supervision_segments.to(torch.int32)

            # get lattice
            lattice = get_lattice(
                nnet_output=nnet_output,
                decoding_graph=self.decoding_graph,
                supervision_segments=supervision_segments,
                search_beam=self.search_beam,
                output_beam=self.output_beam,
                min_active_states=self.min_active_states,
                max_active_states=self.max_active_states,
            )
            lattice_scores = lattice.get_tot_scores(
                log_semiring=self.use_log_semiring,
                use_double_scores=False,
            )
            lattice_loss = -lattice_scores.sum() / batch_size
            return lattice_loss
        else:
            return 0
