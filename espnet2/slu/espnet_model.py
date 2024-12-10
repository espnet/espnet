from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.slu.postdecoder.abs_postdecoder import AbsPostDecoder
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
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


class ESPnetSLUModel(ESPnetASRModel):
    """
    CTC-attention hybrid Encoder-Decoder model for spoken language understanding.

    This model combines the CTC (Connectionist Temporal Classification)
    and attention mechanisms to process and understand spoken language
    inputs. It can handle both speech and text inputs and is suitable
    for tasks such as speech recognition and natural language understanding.

    Attributes:
        blank_id (int): ID for the blank token in CTC.
        sos (int): Start of sequence token ID.
        eos (int): End of sequence token ID.
        vocab_size (int): Size of the vocabulary.
        ignore_id (int): ID of the token to ignore in loss calculations.
        ctc_weight (float): Weight for the CTC loss.
        interctc_weight (float): Weight for the intermediate CTC loss.
        token_list (List[str]): List of tokens.
        transcript_token_list (Optional[List[str]]): List of transcript tokens.
        two_pass (bool): Flag for using two-pass decoding.
        pre_postencoder_norm (bool): Flag for normalization in pre/post-encoder.
        frontend (Optional[AbsFrontend]): Frontend feature extractor.
        specaug (Optional[AbsSpecAug]): SpecAugment module for data augmentation.
        normalize (Optional[AbsNormalize]): Normalization module.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module.
        postdecoder (Optional[AbsPostDecoder]): Post-decoder module.
        encoder (AbsEncoder): Main encoder module.
        decoder (Optional[AbsDecoder]): Decoder module.
        ctc (CTC): CTC module.
        joint_network (Optional[torch.nn.Module]): Joint network for transducer.
        deliberationencoder (Optional[AbsPostEncoder]): Deliberation encoder.
        error_calculator (Optional[ErrorCalculator]): Error calculator for metrics.

    Args:
        vocab_size (int): Size of the vocabulary.
        token_list (Union[Tuple[str, ...], List[str]]): List of tokens.
        frontend (Optional[AbsFrontend]): Frontend feature extractor.
        specaug (Optional[AbsSpecAug]): SpecAugment module.
        normalize (Optional[AbsNormalize]): Normalization module.
        preencoder (Optional[AbsPreEncoder]): Pre-encoder module.
        encoder (AbsEncoder): Encoder module.
        postencoder (Optional[AbsPostEncoder]): Post-encoder module.
        decoder (AbsDecoder): Decoder module.
        ctc (CTC): CTC module.
        joint_network (Optional[torch.nn.Module]): Joint network.
        postdecoder (Optional[AbsPostDecoder]): Post-decoder module.
        deliberationencoder (Optional[AbsPostEncoder]): Deliberation encoder.
        transcript_token_list (Union[Tuple[str, ...], List[str], None], optional):
            List of transcript tokens.
        ctc_weight (float, optional): Weight for CTC loss (default: 0.5).
        interctc_weight (float, optional): Weight for intermediate CTC loss (default: 0.0).
        ignore_id (int, optional): ID to ignore in loss calculations (default: -1).
        lsm_weight (float, optional): Label smoothing weight (default: 0.0).
        length_normalized_loss (bool, optional): Flag for length normalization (default: False).
        report_cer (bool, optional): Flag to report CER (default: True).
        report_wer (bool, optional): Flag to report WER (default: True).
        sym_space (str, optional): Symbol for space (default: "<space>").
        sym_blank (str, optional): Symbol for blank (default: "<blank>").
        extract_feats_in_collect_stats (bool, optional): Flag to extract features
            in statistics collection (default: True).
        two_pass (bool, optional): Flag for two-pass decoding (default: False).
        pre_postencoder_norm (bool, optional): Flag for normalization (default: False).

    Returns:
        None: Initializes the model parameters.

    Examples:
        >>> model = ESPnetSLUModel(
        ...     vocab_size=100,
        ...     token_list=["<blank>", "<space>", "hello", "world"],
        ...     frontend=None,
        ...     specaug=None,
        ...     normalize=None,
        ...     preencoder=None,
        ...     encoder=my_encoder,
        ...     postencoder=None,
        ...     decoder=my_decoder,
        ...     ctc=my_ctc,
        ...     joint_network=None
        ... )
        >>> output = model.forward(speech_tensor, speech_lengths, text_tensor, text_lengths)

    Note:
        The model can be used in both training and inference modes. Ensure to
        appropriately set the training flags when using the model.

    Raises:
        AssertionError: If `ctc_weight` or `interctc_weight` is out of range.
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
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        postdecoder: Optional[AbsPostDecoder] = None,
        deliberationencoder: Optional[AbsPostEncoder] = None,
        transcript_token_list: Union[Tuple[str, ...], List[str], None] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        two_pass: bool = False,
        pre_postencoder_norm: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        AbsESPnetModel.__init__(self)
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()
        if transcript_token_list is not None:
            self.transcript_token_list = transcript_token_list.copy()
        self.two_pass = two_pass
        self.pre_postencoder_norm = pre_postencoder_norm
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.postdecoder = postdecoder
        self.encoder = encoder
        if self.postdecoder is not None:
            if self.encoder._output_size != self.postdecoder.output_size_dim:
                self.uniform_linear = torch.nn.Linear(
                    self.encoder._output_size, self.postdecoder.output_size_dim
                )

        self.deliberationencoder = deliberationencoder
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        transcript: torch.Tensor = None,
        transcript_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a forward pass through the model, including the encoder,
        decoder, and loss calculation.

        This method processes the input speech and text data, passing them
        through the model's encoder and decoder components, and calculates
        the corresponding loss values based on the specified weights for
        CTC and attention losses.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the
                input speech data.
            speech_lengths: A tensor of shape (Batch,) containing the lengths
                of each speech input.
            text: A tensor of shape (Batch, Length) representing the target
                text sequences.
            text_lengths: A tensor of shape (Batch,) containing the lengths
                of each text input.
            transcript: (Optional) A tensor representing additional transcript
                information. Defaults to None.
            transcript_lengths: (Optional) A tensor of lengths for the
                transcripts. Defaults to None.
            kwargs: Additional keyword arguments, where "utt_id" is among the
                inputs.

        Returns:
            A tuple containing:
                - loss: A tensor representing the total loss computed.
                - stats: A dictionary containing various statistics such as
                    loss values and error rates.
                - weight: A tensor representing the batch size or weight for
                    loss computation.

        Raises:
            AssertionError: If the input dimensions do not match or if any
                of the assertions regarding tensor shapes fail.

        Examples:
            >>> model = ESPnetSLUModel(...)
            >>> speech_data = torch.randn(32, 16000)  # Batch of 32 samples
            >>> speech_lengths = torch.tensor([16000] * 32)  # All lengths 16000
            >>> text_data = torch.randint(0, 100, (32, 20))  # Batch of texts
            >>> text_lengths = torch.tensor([20] * 32)  # All lengths 20
            >>> loss, stats, weight = model.forward(
            ...     speech_data, speech_lengths, text_data, text_lengths
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
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, transcript, transcript_lengths
        )
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
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

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        transcript: torch.Tensor = None,
        transcript_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from the input speech tensor.

        This method processes the input speech tensor to extract relevant
        features and their corresponding lengths, returning them in a
        dictionary format. It is typically used in the context of speech
        recognition tasks to prepare input data for the model.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing
                the speech data.
            speech_lengths: A tensor of shape (Batch,) containing the
                lengths of the speech sequences.
            text: A tensor of shape (Batch, Length) representing the
                corresponding text data (not used in feature extraction).
            text_lengths: A tensor of shape (Batch,) containing the
                lengths of the text sequences (not used in feature
                extraction).
            transcript: An optional tensor representing the transcript
                (default: None).
            transcript_lengths: An optional tensor representing the lengths
                of the transcripts (default: None).
            kwargs: Additional keyword arguments for future extension.

        Returns:
            A dictionary containing:
                - "feats": The extracted features tensor.
                - "feats_lengths": The lengths of the extracted features.

        Examples:
            >>> model = ESPnetSLUModel(...)
            >>> speech_tensor = torch.randn(4, 16000)  # 4 samples, 1 second each
            >>> speech_lengths = torch.tensor([16000, 15000, 14000, 13000])
            >>> text_tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
            >>> text_lengths = torch.tensor([3, 3, 3, 3])
            >>> result = model.collect_feats(speech_tensor, speech_lengths, text_tensor, text_lengths)
            >>> print(result['feats'].shape)
            torch.Size([4, ...])  # Shape depends on the feature extraction method
            >>> print(result['feats_lengths'])
            tensor([...])  # Lengths of the extracted features

        Note:
            This method primarily utilizes the `_extract_feats` function
            to perform the actual feature extraction from the input speech
            data.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        transcript_pad: torch.Tensor = None,
        transcript_pad_lens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the input speech through the frontend and encoder.

        This method performs the following steps:
        1. Extracts features from the input speech.
        2. Applies data augmentation if specified and in training mode.
        3. Normalizes the features.
        4. Passes the features through the pre-encoder (if applicable).
        5. Feeds the processed features into the encoder.
        6. Optionally applies a post-encoder for further processing.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the input
                speech data.
            speech_lengths: A tensor of shape (Batch,) representing the lengths of
                the input speech sequences.
            transcript_pad: (Optional) A tensor for padded transcripts, if provided.
            transcript_pad_lens: (Optional) A tensor representing the lengths of
                the padded transcripts.

        Returns:
            A tuple containing:
                - encoder_out: A tensor of shape (Batch, Length2, Dim2)
                  representing the output from the encoder.
                - encoder_out_lens: A tensor representing the lengths of the
                  encoder outputs.

        Examples:
            >>> model = ESPnetSLUModel(vocab_size=5000, token_list=['<blank>', '<sos>', '<eos>'], ...)
            >>> speech_data = torch.randn(2, 16000)  # Example speech data
            >>> speech_lengths = torch.tensor([16000, 15000])  # Lengths of the examples
            >>> encoder_out, encoder_out_lens = model.encode(speech_data, speech_lengths)

        Note:
            This method is primarily used during inference in ASR tasks.
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats,
                feats_lengths,
            )
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        if self.postdecoder is not None:
            if self.encoder._output_size != self.postdecoder.output_size_dim:
                encoder_out = self.uniform_linear(encoder_out)
            transcript_list = [
                " ".join([self.transcript_token_list[int(k)] for k in k1 if k != -1])
                for k1 in transcript_pad
            ]
            (
                transcript_input_id_features,
                transcript_input_mask_features,
                transcript_segment_ids_feature,
                transcript_position_ids_feature,
                input_id_length,
            ) = self.postdecoder.convert_examples_to_features(transcript_list, 128)
            bert_encoder_out = self.postdecoder(
                torch.LongTensor(transcript_input_id_features).to(device=speech.device),
                torch.LongTensor(transcript_input_mask_features).to(
                    device=speech.device
                ),
                torch.LongTensor(transcript_segment_ids_feature).to(
                    device=speech.device
                ),
                torch.LongTensor(transcript_position_ids_feature).to(
                    device=speech.device
                ),
            )
            bert_encoder_lens = torch.LongTensor(input_id_length).to(
                device=speech.device
            )
            bert_encoder_out = bert_encoder_out[:, : torch.max(bert_encoder_lens)]
            final_encoder_out_lens = encoder_out_lens + bert_encoder_lens
            max_lens = torch.max(final_encoder_out_lens)
            encoder_new_out = torch.zeros(
                (encoder_out.shape[0], max_lens, encoder_out.shape[2])
            ).to(device=speech.device)
            for k in range(len(encoder_out)):
                encoder_new_out[k] = torch.cat(
                    (
                        encoder_out[k, : encoder_out_lens[k]],
                        bert_encoder_out[k, : bert_encoder_lens[k]],
                        torch.zeros(
                            (max_lens - final_encoder_out_lens[k], encoder_out.shape[2])
                        ).to(device=speech.device),
                    ),
                    0,
                )
            if self.deliberationencoder is not None:
                encoder_new_out, final_encoder_out_lens = self.deliberationencoder(
                    encoder_new_out, final_encoder_out_lens
                )
            encoder_out = encoder_new_out
            encoder_out_lens = final_encoder_out_lens

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )
        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens
