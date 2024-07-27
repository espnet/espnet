import itertools
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel as SingleESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class PITLossWrapper(AbsLossWrapper):
    """
        Permutation Invariant Training (PIT) Loss Wrapper.

    This class implements a wrapper for applying Permutation Invariant Training
    to a given loss criterion function. It is designed to handle multi-speaker
    scenarios where the order of speakers in the inference and reference may not
    be consistent.

    Attributes:
        criterion_fn (Callable): The loss criterion function to be wrapped.
        num_ref (int): The number of reference speakers.

    Args:
        criterion_fn (Callable): The loss criterion function to be wrapped.
        num_ref (int): The number of reference speakers.

    Note:
        This wrapper is similar to the PIT solver in espnet2/enh/loss/wrapper/pit_solver.py.

    Examples:
        >>> criterion = torch.nn.MSELoss()
        >>> pit_wrapper = PITLossWrapper(criterion, num_ref=2)
        >>> inf = torch.randn(4, 2, 10)  # (batch, num_speakers, features)
        >>> ref = torch.randn(4, 2, 10)  # (batch, num_speakers, features)
        >>> inf_lens = torch.full((4, 2), 10)
        >>> ref_lens = torch.full((4, 2), 10)
        >>> loss, perm = pit_wrapper(inf, inf_lens, ref, ref_lens)
    """

    def __init__(self, criterion_fn: Callable, num_ref: int):
        super().__init__()
        self.criterion_fn = criterion_fn
        self.num_ref = num_ref

    def forward(
        self,
        inf: torch.Tensor,
        inf_lens: torch.Tensor,
        ref: torch.Tensor,
        ref_lens: torch.Tensor,
        others: Dict = None,
    ):
        """
                Apply Permutation Invariant Training (PIT) loss calculation.

        This method computes the PIT loss by finding the optimal permutation of
        speakers that minimizes the total loss across all possible permutations.

        Args:
            inf (torch.Tensor): Inferred output tensor.
                Shape: (batch, num_inf, ...)
            inf_lens (torch.Tensor): Lengths of inferred outputs.
                Shape: (batch, num_inf)
            ref (torch.Tensor): Reference tensor.
                Shape: (batch, num_ref, ...)
            ref_lens (torch.Tensor): Lengths of reference outputs.
                Shape: (batch, num_ref)
            others (Dict, optional): Additional arguments to be passed to the criterion function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The mean of the minimum losses across the batch (scalar).
                - The optimal permutation for each item in the batch.
                  Shape: (batch_size, num_ref)

        Raises:
            AssertionError: If the shapes of input tensors are inconsistent or
                            if num_ref doesn't match the number of speakers in the inputs.

        Note:
            This method assumes that the number of inferred speakers (num_inf) is equal
            to the number of reference speakers (num_ref).

        Examples:
            >>> pit_wrapper = PITLossWrapper(criterion_fn, num_ref=2)
            >>> inf = torch.randn(4, 2, 10)  # (batch, num_speakers, features)
            >>> ref = torch.randn(4, 2, 10)  # (batch, num_speakers, features)
            >>> inf_lens = torch.full((4, 2), 10)
            >>> ref_lens = torch.full((4, 2), 10)
            >>> loss, perm = pit_wrapper.forward(inf, inf_lens, ref, ref_lens)
        """
        assert (
            self.num_ref
            == inf.shape[1]
            == inf_lens.shape[1]
            == ref.shape[1]
            == ref_lens.shape[1]
        ), (self.num_ref, inf.shape, inf_lens.shape, ref.shape, ref_lens.shape)

        all_permutations = torch.as_tensor(
            list(itertools.permutations(range(self.num_ref), r=self.num_ref))
        )

        stats = defaultdict(list)

        def pre_hook(func, *args, **kwargs):
            ret = func(*args, **kwargs)
            for k, v in getattr(self.criterion_fn, "stats", {}).items():
                stats[k].append(v)
            return ret

        def pair_loss(permutation):
            return sum(
                [
                    pre_hook(
                        self.criterion_fn,
                        inf[:, j],
                        inf_lens[:, j],
                        ref[:, i],
                        ref_lens[:, i],
                    )
                    for i, j in enumerate(permutation)
                ]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in all_permutations], dim=1
        )  # (batch_size, num_perm)

        min_losses, min_ids = torch.min(losses, dim=1)
        min_ids = min_ids.cpu()  # because all_permutations is a cpu tensor.
        opt_perm = all_permutations[min_ids]  # (batch_size, num_ref)

        # Permute the inf and inf_lens according to the optimal perm
        return min_losses.mean(), opt_perm

    @classmethod
    def permutate(self, perm, *args):
        """
                Permute the input tensors according to the given permutation.

        This class method applies the optimal permutation to the input tensors,
        rearranging the speaker order for each item in the batch.

        Args:
            perm (torch.Tensor): The permutation tensor to apply.
                Shape: (batch_size, num_ref)
            *args: Variable length argument list of tensors to be permuted.
                Each tensor should have shape (batch, num_inf, ...)

        Returns:
            List[torch.Tensor]: A list of permuted tensors, each with the same shape
            as its corresponding input tensor.

        Raises:
            AssertionError: If the batch size or number of speakers is inconsistent
                            across the input tensors.

        Note:
            This method is typically used after finding the optimal permutation
            with the forward method to reorder the input tensors accordingly.

        Examples:
            >>> pit_wrapper = PITLossWrapper(criterion_fn, num_ref=2)
            >>> inf = torch.randn(4, 2, 10)  # (batch, num_speakers, features)
            >>> inf_lens = torch.full((4, 2), 10)
            >>> perm = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
            >>> permuted_inf, permuted_inf_lens = PITLossWrapper.permutate(perm, inf, inf_lens)
        """
        ret = []
        batch_size = None
        num_ref = None

        for arg in args:  # (batch, num_inf, ...)
            if batch_size is None:
                batch_size, num_ref = arg.shape[:2]
            else:
                assert torch.Size([batch_size, num_ref]) == arg.shape[:2]

            ret.append(
                torch.stack(
                    [arg[torch.arange(batch_size), perm[:, i]] for i in range(num_ref)],
                    dim=1,
                )
            )
        return ret


class ESPnetASRModel(SingleESPnetASRModel):
    """
        CTC-attention hybrid Encoder-Decoder model for multi-speaker ASR.

    This class extends the SingleESPnetASRModel to support multi-speaker automatic speech
    recognition using Permutation Invariant Training (PIT). It combines CTC and
    attention-based approaches for improved ASR performance in scenarios with multiple speakers.

    Attributes:
        num_inf (int): Number of inferences (outputs) from the model.
        num_ref (int): Number of references (ground truth sequences).
        pit_ctc (PITLossWrapper): PIT wrapper for the CTC loss.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend processing module.
        specaug (Optional[AbsSpecAug]): SpecAugment module.
        normalize (Optional[AbsNormalize]): Normalization module.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        encoder (AbsEncoder): Encoder module.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module.
        decoder (Optional[AbsDecoder]): Decoder module.
        ctc (CTC): CTC module.
        joint_network (Optional[torch.nn.Module]): Joint network for transducer-based models.
        ctc_weight (float): Weight of CTC loss (0.0 < ctc_weight <= 1.0).
        interctc_weight (float): Weight of intermediate CTC loss (must be 0.0 for multi-speaker ASR).
        ignore_id (int): Padding value for the ignore index.
        lsm_weight (float): Label smoothing weight.
        length_normalized_loss (bool): Whether to normalize loss by length.
        report_cer (bool): Whether to report Character Error Rate.
        report_wer (bool): Whether to report Word Error Rate.
        sym_space (str): Space symbol.
        sym_blank (str): Blank symbol.
        sym_sos (str): Start of sequence symbol.
        sym_eos (str): End of sequence symbol.
        extract_feats_in_collect_stats (bool): Whether to extract features in collect_stats.
        lang_token_id (int): Language token ID.
        num_inf (int): Number of inferences (outputs) from the model.
        num_ref (int): Number of references (ground truth sequences).

    Note:
        - This model requires that num_inf == num_ref.
        - The interctc_weight must be set to 0.0 as intermediate CTC is not supported for multi-speaker ASR.

    Examples:
        >>> model = ESPnetASRModel(
        ...     vocab_size=1000,
        ...     token_list=["<blank>", "<unk>", "a", "b", "c", ...],
        ...     frontend=frontend,
        ...     specaug=specaug,
        ...     normalize=normalize,
        ...     preencoder=preencoder,
        ...     encoder=encoder,
        ...     postencoder=postencoder,
        ...     decoder=decoder,
        ...     ctc=ctc,
        ...     joint_network=None,
        ...     ctc_weight=0.3,
        ...     interctc_weight=0.0,
        ...     ignore_id=-1,
        ...     lsm_weight=0.1,
        ...     length_normalized_loss=False,
        ...     report_cer=True,
        ...     report_wer=True,
        ...     sym_space="<space>",
        ...     sym_blank="<blank>",
        ...     sym_sos="<sos/eos>",
        ...     sym_eos="<sos/eos>",
        ...     extract_feats_in_collect_stats=True,
        ...     lang_token_id=-1,
        ...     num_inf=2,
        ...     num_ref=2
        ... )
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        # num_inf: the number of inferences (= number of outputs of the model)
        # num_ref: the number of references (= number of groundtruth seqs)
        num_inf: int = 1,
        num_ref: int = 1,
    ):
        assert 0.0 < ctc_weight <= 1.0, ctc_weight
        assert interctc_weight == 0.0, "interctc is not supported for multispeaker ASR"

        super(ESPnetASRModel, self).__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        assert num_inf == num_ref, "Current PIT loss wrapper requires num_inf=num_ref"
        self.num_inf = num_inf
        self.num_ref = num_ref

        self.pit_ctc = PITLossWrapper(criterion_fn=self.ctc, num_ref=num_ref)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
                Forward pass of the ESPnetASRModel for multi-speaker ASR.

        This method processes the input speech and computes the loss for multi-speaker
        automatic speech recognition using a combination of CTC and attention-based approaches.

        Args:
            speech (torch.Tensor): Input speech tensor (Batch, Length, ...).
            speech_lengths (torch.Tensor): Lengths of input speech sequences (Batch,).
            text (torch.Tensor): Target text tensor for the first speaker (Batch, Length).
            text_lengths (torch.Tensor): Lengths of target text sequences (Batch,).
            **kwargs: Additional keyword arguments.
                Expected to contain "text_spk{n}" and "text_spk{n}_lengths" for n = 2 to num_ref,
                representing the text and text lengths for additional speakers.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - loss: The total loss for the batch.
                - stats: A dictionary containing various statistics and metrics:
                    - loss: Total loss (detached).
                    - loss_att: Attention loss (if applicable, detached).
                    - loss_ctc: CTC loss (if applicable, detached).
                    - loss_transducer: Transducer loss (if applicable, detached).
                    - acc: Attention accuracy (if applicable).
                    - cer: Character Error Rate (if applicable).
                    - wer: Word Error Rate (if applicable).
                    - cer_ctc: Character Error Rate for CTC (if applicable).
                    - cer_transducer: Character Error Rate for Transducer (if applicable).
                    - wer_transducer: Word Error Rate for Transducer (if applicable).
                - weight: Batch size (used for averaging).

        Raises:
            AssertionError: If the input tensor dimensions are inconsistent or if the
                            batch sizes don't match across inputs.

        Note:
            - This method handles both CTC and attention-based (and potentially transducer-based) ASR approaches.
            - It uses Permutation Invariant Training (PIT) for handling multiple speakers.
            - The method expects additional text inputs for each speaker beyond the first,
              which should be provided in the kwargs as "text_spk{n}" and "text_spk{n}_lengths".

        Examples:
            >>> speech = torch.randn(2, 1000, 80)  # (batch, time, features)
            >>> speech_lengths = torch.tensor([1000, 800])
            >>> text = torch.randint(0, 100, (2, 50))  # (batch, max_text_len)
            >>> text_lengths = torch.tensor([45, 30])
            >>> text_spk2 = torch.randint(0, 100, (2, 40))
            >>> text_spk2_lengths = torch.tensor([35, 25])
            >>> loss, stats, weight = model.forward(
            ...     speech, speech_lengths, text, text_lengths,
            ...     text_spk2=text_spk2, text_spk2_lengths=text_spk2_lengths
            ... )
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text_ref = [text] + [
            kwargs["text_spk{}".format(spk + 1)] for spk in range(1, self.num_ref)
        ]
        text_ref_lengths = [text_lengths] + [
            kwargs.get("text_spk{}_lengths".format(spk + 1), None)
            for spk in range(1, self.num_ref)
        ]

        assert all(ref_lengths.dim() == 1 for ref_lengths in text_ref_lengths), (
            ref_lengths.shape for ref_lengths in text_ref_lengths
        )

        text_lengths = torch.stack(text_ref_lengths, dim=1)  # (batch, num_ref)
        text_length_max = text_lengths.max()
        # pad text sequences of different speakers to the same length
        text = torch.stack(
            [
                torch.nn.functional.pad(
                    ref, (0, text_length_max - ref.shape[1]), value=self.ignore_id
                )
                for ref in text_ref
            ],
            dim=1,
        )  # (batch, num_ref, seq_len)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            # CTC is computed twice
            # This 1st ctc calculation is only used to decide permutation
            _, perm = self.pit_ctc(encoder_out, encoder_out_lens, text, text_lengths)
            encoder_out, encoder_out_lens = PITLossWrapper.permutate(
                perm, encoder_out, encoder_out_lens
            )
            if text.dim() == 3:  # combine all speakers hidden vectors and labels.
                encoder_out = encoder_out.reshape(-1, *encoder_out.shape[2:])
                encoder_out_lens = encoder_out_lens.reshape(-1)
                text = text.reshape(-1, text.shape[-1])
                text_lengths = text_lengths.reshape(-1)

            # This 2nd ctc calculation is to compute the loss
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            loss_ctc = loss_ctc.sum()

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
