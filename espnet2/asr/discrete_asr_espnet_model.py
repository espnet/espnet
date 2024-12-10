from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.mt.espnet_model import ESPnetMTModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetDiscreteASRModel(ESPnetMTModel):
    """
        ESPnetDiscreteASRModel is an encoder-decoder model for automatic speech
    recognition (ASR) that leverages discrete tokens. It integrates various
    components such as frontend, encoder, decoder, and optionally CTC for
    improved performance.

    Attributes:
        vocab_size (int): The size of the vocabulary used for decoding.
        token_list (List[str]): A list of tokens corresponding to the vocabulary.
        frontend (AbsFrontend): An optional frontend for feature extraction.
        specaug (AbsSpecAug): An optional data augmentation technique.
        preencoder (AbsPreEncoder): An optional preencoder for raw input data.
        encoder (AbsEncoder): The encoder component of the model.
        postencoder (AbsPostEncoder): An optional postencoder for additional processing.
        decoder (AbsDecoder): The decoder component of the model.
        ctc (CTC): An optional CTC module for training.
        ctc_weight (float): Weight for the CTC loss in the combined loss function.
        interctc_weight (float): Weight for the intermediate CTC loss.
        ignore_id (int): Token ID to ignore in the loss calculation.
        length_normalized_loss (bool): If True, normalize the loss by length.
        report_bleu (bool): If True, report BLEU score during training.
        sym_space (str): Symbol representing space in the token list.
        sym_blank (str): Symbol representing blank in the token list.
        blank_id (int): The ID of the blank token.
        error_calculator (ASRErrorCalculator): Calculates error metrics like CER and WER.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend component.
        specaug (Optional[AbsSpecAug]): SpecAugment component.
        preencoder (Optional[AbsPreEncoder]): Preencoder component.
        encoder (AbsEncoder): Encoder component.
        postencoder (Optional[AbsPostEncoder]): Postencoder component.
        decoder (AbsDecoder): Decoder component.
        ctc (Optional[CTC]): CTC component.
        ctc_weight (float): Weight for CTC loss (default 0.5).
        interctc_weight (float): Weight for intermediate CTC loss (default 0.0).
        src_vocab_size (int): Source vocabulary size (default 0).
        src_token_list (Union[Tuple[str, ...], List[str]]): Source token list (default []).
        ignore_id (int): ID to ignore in loss calculation (default -1).
        lsm_weight (float): Label smoothing weight (default 0.0).
        length_normalized_loss (bool): Normalize loss by length (default False).
        report_bleu (bool): Report BLEU score (default True).
        sym_space (str): Symbol for space (default "<space>").
        sym_blank (str): Symbol for blank (default "<blank>").
        patch_size (int): Patch size for model (default 1).
        extract_feats_in_collect_stats (bool): Extract features during statistics collection (default True).
        share_decoder_input_output_embed (bool): Share decoder input/output embedding (default False).
        share_encoder_decoder_input_embed (bool): Share encoder/decoder input embedding (default False).

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the
        total loss, a dictionary of statistics, and the batch size.

    Raises:
        AssertionError: If the input lengths are inconsistent or invalid.

    Examples:
        model = ESPnetDiscreteASRModel(
            vocab_size=5000,
            token_list=["<blank>", "<space>", "hello", "world"],
            frontend=None,
            specaug=None,
            preencoder=None,
            encoder=my_encoder,
            postencoder=None,
            decoder=my_decoder,
            ctc=my_ctc,
        )

        loss, stats, batch_size = model(
            text=torch.randint(0, 5000, (32, 10)),
            text_lengths=torch.randint(1, 11, (32,)),
            src_text=torch.randint(0, 5000, (32, 10)),
            src_text_lengths=torch.randint(1, 11, (32,))
        )

    Note:
        This model supports optional components for various stages of processing,
        allowing for flexibility in architecture design.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: Optional[CTC],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        src_vocab_size: int = 0,
        src_token_list: Union[Tuple[str, ...], List[str]] = [],
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        patch_size: int = 1,
        extract_feats_in_collect_stats: bool = True,
        share_decoder_input_output_embed: bool = False,
        share_encoder_decoder_input_embed: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            src_vocab_size=src_vocab_size,
            src_token_list=src_token_list,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_bleu=report_bleu,
            sym_space=sym_space,
            sym_blank=sym_blank,
            patch_size=patch_size,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            share_decoder_input_output_embed=share_decoder_input_output_embed,
            share_encoder_decoder_input_embed=share_encoder_decoder_input_embed,
        )

        self.specaug = specaug
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        if report_bleu:
            self.error_calculator = ASRErrorCalculator(
                token_list, sym_space, sym_blank, True, True
            )

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Decoder + Calculate loss.

        This method performs the forward pass through the entire model,
        processing the input text through the frontend, encoder, and decoder
        while also calculating the associated loss.

        Args:
            text: Tensor of shape (Batch, Length) representing the target
                sequences.
            text_lengths: Tensor of shape (Batch,) containing the lengths of
                each target sequence.
            src_text: Tensor of shape (Batch, Length) representing the source
                sequences.
            src_text_lengths: Tensor of shape (Batch,) containing the lengths of
                each source sequence.
            kwargs: Additional keyword arguments, where "utt_id" may be included.

        Returns:
            A tuple containing:
                - loss: A tensor representing the computed loss.
                - stats: A dictionary of various statistics computed during
                  the forward pass.
                - weight: A tensor representing the batch size.

        Raises:
            AssertionError: If the dimensions of input tensors do not match.

        Examples:
            >>> model = ESPnetDiscreteASRModel(...)
            >>> text = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> text_lengths = torch.tensor([3, 3])
            >>> src_text = torch.tensor([[1, 2], [3, 4]])
            >>> src_text_lengths = torch.tensor([2, 2])
            >>> loss, stats, weight = model.forward(text, text_lengths, src_text,
            ...                                      src_text_lengths)

        Note:
            Ensure that the input tensors are properly padded and
            that the lengths are accurately specified.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
            == src_text.shape[0]
            == src_text_lengths.shape[0]
        ), (text.shape, text_lengths.shape, src_text.shape, src_text_lengths.shape)

        batch_size = src_text.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        src_text = src_text[:, : src_text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(src_text, src_text_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        # 2a. Attention-decoder branch (MT)
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. Loss computation
        if self.ctc_weight > 0.0:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        else:
            loss = loss_att

        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self, src_text: torch.Tensor, src_text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Frontend + Encoder. Note that this method is used by mt_inference.py.

        This method processes the input source text through a series of layers
        including the frontend, preencoder (if applicable), and the main encoder
        to produce the encoded output and its lengths.

        Args:
            src_text: A tensor of shape (Batch, Length, ...), representing the
                input source text sequences.
            src_text_lengths: A tensor of shape (Batch,), containing the lengths
                of the source text sequences.

        Returns:
            A tuple containing:
                - encoder_out: A tensor of shape (Batch, Length2, Dim2), representing
                  the encoded output from the encoder.
                - encoder_out_lens: A tensor of shape (Batch,), containing the lengths
                  of the encoded output sequences.

        Note:
            - This method assumes that the input text has already been processed
              to a suitable format for encoding.
            - The method can perform data augmentation if the model is in training
              mode and a spec augmentation instance is provided.

        Examples:
            >>> model = ESPnetDiscreteASRModel(...)
            >>> src_text = torch.randn(2, 10, 256)  # Example input tensor
            >>> src_text_lengths = torch.tensor([10, 8])  # Lengths of the inputs
            >>> encoder_out, encoder_out_lens = model.encode(src_text, src_text_lengths)
            >>> print(encoder_out.shape)  # Expected shape: (2, Length2, Dim2)
            >>> print(encoder_out_lens)  # Lengths of the encoded sequences
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(src_text, src_text_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        # encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == src_text.size(0), (
            encoder_out.size(),
            src_text.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_mt(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
