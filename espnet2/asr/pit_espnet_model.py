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
    Wrapper for Permutation Invariant Training (PIT) loss.

    This class wraps a given loss function to compute the permutation invariant 
    loss for multi-reference scenarios. It takes multiple inferences and 
    references, calculates all possible permutations, and returns the minimum 
    loss along with the optimal permutation.

    Attributes:
        criterion_fn (Callable): The loss function to be used for computing 
            the loss for each reference-inference pair.
        num_ref (int): The number of reference signals.

    Args:
        criterion_fn (Callable): A callable loss function that takes 
            inference and reference tensors along with their lengths.
        num_ref (int): The number of reference signals for the PIT loss 
            computation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean 
        minimum loss across all batches and the optimal permutation indices.

    Raises:
        AssertionError: If the dimensions of the input tensors do not match 
        the expected shapes.

    Examples:
        >>> import torch
        >>> criterion = SomeLossFunction()  # Replace with an actual loss function
        >>> pit_loss_wrapper = PITLossWrapper(criterion_fn=criterion, num_ref=2)
        >>> inf = torch.randn(5, 2, 10)  # (batch, num_inf, features)
        >>> inf_lens = torch.tensor([10] * 5)  # (batch,)
        >>> ref = torch.randn(5, 2, 10)  # (batch, num_ref, features)
        >>> ref_lens = torch.tensor([10] * 5)  # (batch,)
        >>> loss, opt_perm = pit_loss_wrapper(inf, inf_lens, ref, ref_lens)
        >>> print(loss, opt_perm)

    Note:
        The loss function used should be capable of handling the inputs as 
        defined in the `forward` method. Ensure that the number of 
        inferences equals the number of references for proper functionality.
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
        Computes the Permutation Invariant Training (PIT) loss using a provided 
        criterion function for multiple references.

        This method takes in inference and reference tensors, along with their 
        lengths, and computes the optimal permutation of the references to 
        minimize the loss. The PIT loss is particularly useful in scenarios 
        where the order of references may vary, such as in speech separation tasks.

        Args:
            inf (torch.Tensor): Inference tensor of shape (batch, num_inf, ...).
            inf_lens (torch.Tensor): Lengths of the inference tensors, shape 
                (batch, num_inf).
            ref (torch.Tensor): Reference tensor of shape (batch, num_ref, ...).
            ref_lens (torch.Tensor): Lengths of the reference tensors, shape 
                (batch, num_ref).
            others (Dict, optional): Additional keyword arguments that may be 
                required by the criterion function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The mean of the minimum losses across the batch.
                - The optimal permutation of the references as a tensor of shape 
                (batch_size, num_ref).

        Raises:
            AssertionError: If the number of references does not match the shapes 
            of the input tensors.

        Examples:
            >>> inf = torch.rand(2, 3, 10)  # Example inference
            >>> inf_lens = torch.tensor([[10, 9, 8], [10, 10, 10]])
            >>> ref = torch.rand(2, 2, 10)  # Example references
            >>> ref_lens = torch.tensor([[10, 9], [10, 10]])
            >>> pit_loss_wrapper = PITLossWrapper(criterion_fn=some_loss_function, 
            ...                                    num_ref=2)
            >>> loss, optimal_perm = pit_loss_wrapper.forward(inf, inf_lens, ref, ref_lens)
            >>> print(loss, optimal_perm)

        Note:
            Ensure that the number of references (num_ref) specified during 
            initialization matches the second dimension of the inference and 
            reference tensors.
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
    ESPnetASRModel is a hybrid CTC-attention Encoder-Decoder model for automatic
    speech recognition (ASR). This model combines the strengths of Connectionist
    Temporal Classification (CTC) and attention mechanisms, enabling it to handle
    different types of input sequences effectively.

    Attributes:
        num_inf (int): The number of inferences (outputs) from the model.
        num_ref (int): The number of references (ground truth sequences).
        pit_ctc (PITLossWrapper): A wrapper for calculating the Permutation Invariant 
            Training (PIT) loss with CTC.

    Args:
        vocab_size (int): The size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens in the 
            vocabulary.
        frontend (Optional[AbsFrontend]): Frontend processing component.
        specaug (Optional[AbsSpecAug]): SpecAugment component for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization layer.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder component.
        encoder (AbsEncoder): The encoder component of the model.
        postencoder (Optional[AbsPostEncoder]): Post-encoder component.
        decoder (Optional[AbsDecoder]): The decoder component of the model.
        ctc (CTC): The CTC component for loss calculation.
        joint_network (Optional[torch.nn.Module]): Joint network component.
        ctc_weight (float): Weight for the CTC loss in the combined loss. Defaults to 0.5.
        interctc_weight (float): Weight for the inter-CTC loss. Defaults to 0.0.
        ignore_id (int): The token ID to ignore during loss calculation. Defaults to -1.
        lsm_weight (float): Label smoothing weight. Defaults to 0.0.
        length_normalized_loss (bool): Whether to use length-normalized loss. Defaults to False.
        report_cer (bool): Whether to report Character Error Rate (CER). Defaults to True.
        report_wer (bool): Whether to report Word Error Rate (WER). Defaults to True.
        sym_space (str): Symbol for space token. Defaults to "<space>".
        sym_blank (str): Symbol for blank token in CTC. Defaults to "<blank>".
        sym_sos (str): Symbol for start of sequence. Defaults to "<sos/eos>".
        sym_eos (str): Symbol for end of sequence. Defaults to "<sos/eos>".
        extract_feats_in_collect_stats (bool): Whether to extract features in 
            collecting statistics. Defaults to True.
        lang_token_id (int): Language token ID. Defaults to -1.
        num_inf (int): Number of inferences (outputs) from the model. Defaults to 1.
        num_ref (int): Number of references (ground truth sequences). Defaults to 1.

    Raises:
        AssertionError: If `ctc_weight` is not in the range (0.0, 1.0] or if 
            `interctc_weight` is not equal to 0.0.
        AssertionError: If `num_inf` is not equal to `num_ref`.

    Examples:
        # Create an instance of ESPnetASRModel
        model = ESPnetASRModel(
            vocab_size=1000,
            token_list=["<blank>", "<space>", "<sos/eos>"] + list("abcdefghijklmnopqrstuvwxyz"),
            frontend=None,
            specaug=None,
            normalize=None,
            preencoder=None,
            encoder=my_encoder,
            postencoder=None,
            decoder=my_decoder,
            ctc=my_ctc,
            joint_network=None,
            ctc_weight=0.5,
            interctc_weight=0.0,
            ignore_id=-1,
            lsm_weight=0.0,
            length_normalized_loss=False,
            report_cer=True,
            report_wer=True,
            sym_space="<space>",
            sym_blank="<blank>",
            sym_sos="<sos/eos>",
            sym_eos="<sos/eos>",
            extract_feats_in_collect_stats=True,
            lang_token_id=-1,
            num_inf=1,
            num_ref=1,
        )

        # Forward pass through the model
        loss, stats, weight = model.forward(
            speech=my_speech_tensor,
            speech_lengths=my_speech_lengths_tensor,
            text=my_text_tensor,
            text_lengths=my_text_lengths_tensor,
            utt_id="example_id"
        )
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
        Forward pass of the ESPnetASRModel, which processes the input speech data
        through the frontend, encoder, and decoder, and calculates the loss based 
        on the provided references. This method supports multiple references for 
        enhanced performance in speech recognition tasks.

        Args:
            speech (torch.Tensor): Input speech tensor of shape (Batch, Length, ...).
            speech_lengths (torch.Tensor): Lengths of the input speech tensor of shape (Batch,).
            text (torch.Tensor): Target text tensor of shape (Batch, Length).
            text_lengths (torch.Tensor): Lengths of the target text tensor of shape (Batch,).
            **kwargs: Additional keyword arguments. Must include "utt_id" and may include 
                references for additional speakers, e.g., "text_spk1", "text_spk1_lengths", etc.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - loss: Computed loss value for the batch.
                - stats: Dictionary containing various statistics from the model, such as 
                  loss values and accuracy metrics.
                - weight: The batch size for further processing.

        Raises:
            AssertionError: If the dimensions of input tensors do not match as expected.

        Examples:
            >>> model = ESPnetASRModel(...)
            >>> speech = torch.randn(4, 100, 80)  # Batch of 4, Length 100, 80 features
            >>> speech_lengths = torch.tensor([100, 90, 80, 70])
            >>> text = torch.randint(0, 30, (4, 20))  # Batch of 4, Length 20
            >>> text_lengths = torch.tensor([20, 18, 15, 12])
            >>> loss, stats, weight = model.forward(speech, speech_lengths, text, text_lengths, 
            ...                                       utt_id='utt1', text_spk1=text, 
            ...                                       text_spk1_lengths=text_lengths)

        Note:
            Ensure that the input tensors are properly padded and have the correct
            dimensions. The `text` and `text_lengths` should match the batch size
            of the `speech` and `speech_lengths` tensors.
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
