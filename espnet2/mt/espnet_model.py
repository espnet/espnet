import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetMTModel(AbsESPnetModel):
    """
    Encoder-Decoder model for machine translation.

    This class implements an encoder-decoder architecture specifically for
    machine translation tasks. It allows for various configurations of
    frontends, encoders, decoders, and post-encoders. The model can also
    compute loss using label smoothing and track BLEU scores during
    evaluation.

    Attributes:
        sos (int): Start of sequence token ID.
        eos (int): End of sequence token ID.
        src_sos (int): Source start of sequence token ID.
        src_eos (int): Source end of sequence token ID.
        vocab_size (int): Size of the target vocabulary.
        src_vocab_size (int): Size of the source vocabulary.
        ignore_id (int): Token ID to ignore during loss computation.
        patch_size (int): Size of the patch for feature extraction.
        token_list (List[str]): List of tokens for BLEU score calculation.
        frontend (AbsFrontend): Frontend module for feature extraction.
        preencoder (AbsPreEncoder): Pre-encoder module.
        postencoder (AbsPostEncoder): Post-encoder module.
        encoder (AbsEncoder): Encoder module.
        decoder (AbsDecoder): Decoder module.
        criterion_mt (LabelSmoothingLoss): Loss function for machine translation.
        mt_error_calculator (MTErrorCalculator): Calculator for BLEU scores.
        extract_feats_in_collect_stats (bool): Flag to extract features.

    Args:
        vocab_size (int): Size of the target vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of target tokens.
        frontend (Optional[AbsFrontend]): Frontend module for feature extraction.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        encoder (AbsEncoder): Encoder module.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module.
        decoder (AbsDecoder): Decoder module.
        src_vocab_size (int, optional): Size of the source vocabulary. Default is 0.
        src_token_list (Union[Tuple[str, ...], List[str]], optional): List of
            source tokens. Default is an empty list.
        ignore_id (int, optional): Token ID to ignore during loss computation.
            Default is -1.
        lsm_weight (float, optional): Weight for label smoothing. Default is 0.0.
        length_normalized_loss (bool, optional): Whether to use length-normalized
            loss. Default is False.
        report_bleu (bool, optional): Whether to report BLEU score. Default is True.
        sym_space (str, optional): Symbol for space. Default is "<space>".
        sym_blank (str, optional): Symbol for blank. Default is "<blank>".
        patch_size (int, optional): Size of the patch for feature extraction.
            Default is 1.
        extract_feats_in_collect_stats (bool, optional): Flag to extract features
            in collect stats. Default is True.
        share_decoder_input_output_embed (bool, optional): Whether to share
            decoder input and output embeddings. Default is False.
        share_encoder_decoder_input_embed (bool, optional): Whether to share
            encoder and decoder input embeddings. Default is False.

    Examples:
        # Create a machine translation model
        model = ESPnetMTModel(
            vocab_size=5000,
            token_list=["<blank>", "<space>", "hello", "world"],
            frontend=None,
            preencoder=None,
            encoder=SomeEncoder(),
            postencoder=None,
            decoder=SomeDecoder(),
            src_vocab_size=3000,
            src_token_list=["<blank>", "<space>", "hola", "mundo"],
            ignore_id=-1,
            lsm_weight=0.1,
            length_normalized_loss=True,
            report_bleu=True,
            sym_space="<space>",
            sym_blank="<blank>",
            patch_size=1,
            extract_feats_in_collect_stats=True,
            share_decoder_input_output_embed=False,
            share_encoder_decoder_input_embed=False,
        )

    Note:
        The `forward` method computes the output and loss for a given input
        batch. The `encode` method extracts features from the source text
        and passes them through the encoder.

    Raises:
        AssertionError: If the input tensor dimensions do not match expected
        dimensions.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
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

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.src_sos = src_vocab_size - 1 if src_vocab_size else None
        self.src_eos = src_vocab_size - 1 if src_vocab_size else None
        self.vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size
        self.ignore_id = ignore_id
        self.patch_size = patch_size
        self.token_list = token_list.copy()

        if share_decoder_input_output_embed:
            if decoder.output_layer is not None:
                decoder.output_layer.weight = decoder.embed[0].weight
                logging.info(
                    "Decoder input embedding and output linear layer are shared"
                )
            else:
                logging.warning(
                    "Decoder has no output layer, so it cannot be shared "
                    "with input embedding"
                )

        if share_encoder_decoder_input_embed:
            if src_vocab_size == vocab_size:
                frontend.embed[0].weight = decoder.embed[0].weight
                logging.info("Encoder and decoder input embeddings are shared")
            else:
                logging.warning(
                    f"src_vocab_size ({src_vocab_size}) does not equal tgt_vocab_size"
                    f" ({vocab_size}), so the encoder and decoder input embeddings "
                    "cannot be shared"
                )

        self.frontend = frontend
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self.decoder = decoder

        self.criterion_mt = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # MT error calculator
        if report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list, sym_space, sym_blank, report_bleu
            )
        else:
            self.mt_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Decoder + Calc loss

        This method processes the input through the frontend, encoder, and decoder
        to calculate the loss for the given input text and source text.

        Args:
            text: A tensor of shape (Batch, Length) representing the target text.
            text_lengths: A tensor of shape (Batch,) containing the lengths of each
                sequence in the target text.
            src_text: A tensor of shape (Batch, Length) representing the source text.
            src_text_lengths: A tensor of shape (Batch,) containing the lengths of
                each sequence in the source text.
            kwargs: Additional arguments, where "utt_id" is among the input.

        Returns:
            A tuple containing:
                - loss: A tensor representing the calculated loss.
                - stats: A dictionary with keys 'loss', 'acc', and 'bleu', containing
                    the corresponding statistics.
                - weight: A tensor representing the batch size for gathering.

        Raises:
            AssertionError: If the dimensions of input tensors do not match.

        Examples:
            >>> model = ESPnetMTModel(...)
            >>> text = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> text_lengths = torch.tensor([3, 3])
            >>> src_text = torch.tensor([[7, 8, 9], [10, 11, 12]])
            >>> src_text_lengths = torch.tensor([3, 3])
            >>> loss, stats, weight = model.forward(text, text_lengths, src_text,
            ...                                      src_text_lengths)
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

        # 2a. Attention-decoder branch (MT)
        loss_mt_att, acc_mt_att, bleu_mt_att = self._calc_mt_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. Loss computation
        loss = loss_mt_att

        stats = dict(
            loss=loss.detach(),
            acc=acc_mt_att,
            bleu=bleu_mt_att,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect features from the source text and return them along with their
        lengths.

        This method extracts features from the source text tensor and returns
        them in a dictionary. If the `extract_feats_in_collect_stats` attribute
        is set to `True`, actual features are extracted; otherwise, dummy
        features are generated.

        Args:
            text (torch.Tensor): The target text tensor of shape (Batch, Length).
            text_lengths (torch.Tensor): The lengths of the target text of shape
                (Batch,).
            src_text (torch.Tensor): The source text tensor of shape (Batch, Length).
            src_text_lengths (torch.Tensor): The lengths of the source text of
                shape (Batch,).
            **kwargs: Additional keyword arguments, if any.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "feats" (torch.Tensor): The extracted features.
                - "feats_lengths" (torch.Tensor): The lengths of the extracted
                  features.

        Raises:
            AssertionError: If the `src_text_lengths` tensor does not have a
                dimension of 1.

        Examples:
            >>> model = ESPnetMTModel(vocab_size=1000, token_list=["<blank>", "<sos>", "<eos>"],
            ...                       encoder=encoder, decoder=decoder)
            >>> text = torch.randint(0, 1000, (32, 20))
            >>> text_lengths = torch.randint(1, 21, (32,))
            >>> src_text = torch.randint(0, 1000, (32, 15))
            >>> src_text_lengths = torch.randint(1, 16, (32,))
            >>> feats = model.collect_feats(text, text_lengths, src_text,
            ...                              src_text_lengths)
            >>> print(feats["feats"].shape)  # Output: (Batch, NSamples, Dim)

        Note:
            This method is particularly useful for feature extraction during
            model evaluation or inference.
        """
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(src_text, src_text_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = src_text, src_text_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, src_text: torch.Tensor, src_text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by mt_inference.py

        Args:
            src_text: (Batch, Length, ...)
            src_text_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(src_text, src_text_lengths)

            # 2. Data augmentation
            # if self.specaug is not None and self.training:
            #     feats, feats_lengths = self.specaug(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

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

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, src_text: torch.Tensor, src_text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert src_text_lengths.dim() == 1, src_text_lengths.shape

        # for data-parallel
        src_text = src_text[:, : src_text_lengths.max()]
        src_text, _ = add_sos_eos(
            src_text, self.src_sos, self.src_eos, self.ignore_id, repeat=self.patch_size
        )
        src_text_lengths = src_text_lengths + self.patch_size

        if self.frontend is not None:
            # Frontend
            #  e.g. Embedding Lookup
            # src_text (Batch, NSamples) -> feats: (Batch, NSamples, Dim)
            feats, feats_lengths = self.frontend(src_text, src_text_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = src_text, src_text_lengths
        return feats, feats_lengths

    def _calc_mt_att_loss(
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
        if self.training or self.mt_error_calculator is None:
            bleu_att = None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            bleu_att = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, bleu_att
