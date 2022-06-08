"""CER/WER computation for Transducer model."""

from typing import List, Tuple, Union

import torch

from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.rnn_decoder import RNNDecoder


class ErrorCalculator(object):
    """CER and WER computation for Transducer model.

    Args:
        decoder: Decoder module.
        joint_network: Joint network module.
        token_list: Set of unique labels.
        sym_space: Space symbol.
        sym_blank: Blank symbol.
        report_cer: Whether to compute CER.
        report_wer: Whether to compute WER.

    """

    def __init__(
        self,
        decoder: Union[RNNDecoder, CustomDecoder],
        joint_network: JointNetwork,
        token_list: List[int],
        sym_space: str,
        sym_blank: str,
        report_cer: bool = False,
        report_wer: bool = False,
    ):
        """Construct an ErrorCalculator object for Transducer model."""
        super().__init__()

        self.beam_search = BeamSearchTransducer(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=2,
            search_type="default",
        )

        self.decoder = decoder

        self.token_list = token_list
        self.space = sym_space
        self.blank = sym_blank

        self.report_cer = report_cer
        self.report_wer = report_wer

    def __call__(
        self, enc_out: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        """Calculate sentence-level CER/WER score for hypotheses sequences.

        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)

        Returns:
            cer: Sentence-level CER score.
            wer: Sentence-level WER score.

        """
        cer, wer = None, None

        batchsize = int(enc_out.size(0))
        batch_nbest = []

        enc_out = enc_out.to(next(self.decoder.parameters()).device)

        for b in range(batchsize):
            nbest_hyps = self.beam_search(enc_out[b])
            batch_nbest.append(nbest_hyps[-1])

        batch_nbest = [nbest_hyp.yseq[1:] for nbest_hyp in batch_nbest]

        hyps, refs = self.convert_to_char(batch_nbest, target.cpu())

        if self.report_cer:
            cer = self.calculate_cer(hyps, refs)

        if self.report_wer:
            wer = self.calculate_wer(hyps, refs)

        return cer, wer

    def convert_to_char(
        self, hyps: torch.Tensor, refs: torch.Tensor
    ) -> Tuple[List, List]:
        """Convert label ID sequences to character.

        Args:
            hyps: Hypotheses sequences. (B, L)
            refs: References sequences. (B, L)

        Returns:
            char_hyps: Character list of hypotheses.
            char_hyps: Character list of references.

        """
        char_hyps, char_refs = [], []

        for i, hyp in enumerate(hyps):
            hyp_i = [self.token_list[int(h)] for h in hyp]
            ref_i = [self.token_list[int(r)] for r in refs[i]]

            char_hyp = "".join(hyp_i).replace(self.space, " ")
            char_hyp = char_hyp.replace(self.blank, "")
            char_ref = "".join(ref_i).replace(self.space, " ")

            char_hyps.append(char_hyp)
            char_refs.append(char_ref)

        return char_hyps, char_refs

    def calculate_cer(self, hyps: torch.Tensor, refs: torch.Tensor) -> float:
        """Calculate sentence-level CER score.

        Args:
            hyps: Hypotheses sequences. (B, L)
            refs: References sequences. (B, L)

        Returns:
            : Average sentence-level CER score.

        """
        import editdistance

        distances, lens = [], []

        for i, hyp in enumerate(hyps):
            char_hyp = hyp.replace(" ", "")
            char_ref = refs[i].replace(" ", "")

            distances.append(editdistance.eval(char_hyp, char_ref))
            lens.append(len(char_ref))

        return float(sum(distances)) / sum(lens)

    def calculate_wer(self, hyps: torch.Tensor, refs: torch.Tensor) -> float:
        """Calculate sentence-level WER score.

        Args:
            hyps: Hypotheses sequences. (B, L)
            refs: References sequences. (B, L)

        Returns:
            : Average sentence-level WER score.

        """
        import editdistance

        distances, lens = [], []

        for i, hyp in enumerate(hyps):
            word_hyp = hyp.split()
            word_ref = refs[i].split()

            distances.append(editdistance.eval(word_hyp, word_ref))
            lens.append(len(word_ref))

        return float(sum(distances)) / sum(lens)
