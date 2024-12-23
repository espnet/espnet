import logging
import random
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.st.espnet_model import ESPnetSTModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetEnhS2TModel(AbsESPnetModel):
    """
    Joint model for Enhancement and Speech to Text (S2T).

    This class combines an enhancement model and a speech-to-text model, allowing
    for joint training and inference for tasks involving speech enhancement and
    transcription. It can handle multiple types of speech-to-text models, including
    automatic speech recognition (ASR), speech translation (ST), and speaker
    diarization.

    Attributes:
        enh_model (ESPnetEnhancementModel): The enhancement model used for speech
            enhancement.
        s2t_model (Union[ESPnetASRModel, ESPnetSTModel, ESPnetDiarizationModel]):
            The speech-to-text model used for transcribing enhanced speech.
        bypass_enh_prob (float): Probability of bypassing the enhancement model during
            training.
        calc_enh_loss (bool): Flag indicating whether to calculate enhancement loss.
        extract_feats_in_collect_stats (bool): Flag to determine if features should
            be extracted during statistics collection.

    Args:
        enh_model (ESPnetEnhancementModel): The enhancement model.
        s2t_model (Union[ESPnetASRModel, ESPnetSTModel, ESPnetDiarizationModel]):
            The speech-to-text model (ASR, ST, or DIAR).
        calc_enh_loss (bool): Whether to calculate enhancement loss. Default is True.
        bypass_enh_prob (float): Probability to bypass enhancement. Default is 0.

    Raises:
        NotImplementedError: If the provided speech-to-text model type is not supported.

    Examples:
        >>> enh_model = ESPnetEnhancementModel(...)
        >>> s2t_model = ESPnetASRModel(...)
        >>> model = ESPnetEnhS2TModel(enh_model, s2t_model)
        >>> speech = torch.randn(2, 16000)  # (Batch, Length)
        >>> lengths = torch.tensor([16000, 16000])
        >>> loss, stats, weight = model(speech, speech_lengths=lengths)

    Note:
        The model's `forward` method performs both enhancement and transcription,
        calculating the necessary losses based on the specified configurations.

    Todo:
        - Implement additional logging for debugging and monitoring.
    """

    @typechecked
    def __init__(
        self,
        enh_model: ESPnetEnhancementModel,
        s2t_model: Union[ESPnetASRModel, ESPnetSTModel, ESPnetDiarizationModel],
        calc_enh_loss: bool = True,
        bypass_enh_prob: float = 0,  # 0 means do not bypass enhancement for all data
    ):

        super().__init__()
        self.enh_model = enh_model
        self.s2t_model = s2t_model  # ASR or ST or DIAR model

        self.bypass_enh_prob = bypass_enh_prob

        self.calc_enh_loss = calc_enh_loss
        if isinstance(self.s2t_model, ESPnetDiarizationModel):
            self.extract_feats_in_collect_stats = False
        else:
            self.extract_feats_in_collect_stats = (
                self.s2t_model.extract_feats_in_collect_stats
            )

        if (
            self.enh_model.num_spk is not None
            and self.enh_model.num_spk > 1
            and isinstance(self.s2t_model, ESPnetASRModel)
        ):
            if self.calc_enh_loss:
                logging.warning("The permutation issue will be handled by the Enh loss")
            else:
                logging.warning("The permutation issue will be handled by the CTC loss")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Decoder + Calculate loss.

        This method processes the input speech tensor through the enhancement
        model and the speech-to-text (S2T) model. It computes the necessary
        losses for training and returns them along with relevant statistics.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the
                input speech signals.
            speech_lengths: A tensor of shape (Batch,) indicating the lengths
                of the input speech signals. Default is None for chunk iterators
                since they do not return the speech lengths. See
                espnet2/iterators/chunk_iter_factory.py for more details.
            **kwargs: Additional keyword arguments, which may include:
                - For Enh+ASR task:
                    text_spk1: (Batch, Length) tensor of text sequences for
                        speaker 1.
                    text_spk2: (Batch, Length) tensor of text sequences for
                        speaker 2.
                    ...
                    text_spk1_lengths: (Batch,) tensor of lengths for text
                        sequences of speaker 1.
                    text_spk2_lengths: (Batch,) tensor of lengths for text
                        sequences of speaker 2.
                    ...
                - For other tasks:
                    text: (Batch, Length) tensor of text sequences. Default
                        is None, included to maintain argument order.
                    text_lengths: (Batch,) tensor of lengths for text sequences.
                        Default is None for the same reason as speech_lengths.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - A tensor representing the computed loss.
                - A dictionary containing various statistics related to the
                  forward pass.
                - A tensor representing the weight for the loss.

        Raises:
            NotImplementedError: If the provided S2T model type is not
            supported.

        Examples:
            >>> model = ESPnetEnhS2TModel(enh_model, s2t_model)
            >>> speech_input = torch.randn(2, 16000)  # Example speech input
            >>> lengths = torch.tensor([16000, 16000])  # Lengths of the input
            >>> loss, stats, weight = model.forward(speech_input, lengths,
            ...                                      text_spk1=text1,
            ...                                      text_spk1_lengths=lengths1)

        Note:
            Ensure that the input tensors are properly shaped and that
            the appropriate keyword arguments are passed based on the task
            (Enh+ASR or others).
        """
        if "text" in kwargs:
            text = kwargs["text"]
            text_ref_lengths = [kwargs.get("text_lengths", None)]
            if text_ref_lengths[0] is not None:
                text_length_max = max(
                    ref_lengths.max() for ref_lengths in text_ref_lengths
                )
            else:
                text_length_max = text.shape[1]
        else:
            text_ref = [
                kwargs["text_spk{}".format(spk + 1)]
                for spk in range(self.enh_model.num_spk)
            ]
            text_ref_lengths = [
                kwargs.get("text_spk{}_lengths".format(spk + 1), None)
                for spk in range(self.enh_model.num_spk)
            ]

            # for data-parallel
            if text_ref_lengths[0] is not None:
                text_length_max = max(
                    ref_lengths.max() for ref_lengths in text_ref_lengths
                )
            else:
                text_length_max = max(text.shape[1] for text in text_ref)
            # pad text sequences of different speakers to the same length
            ignore_id = getattr(self.s2t_model, "ignore_id", -1)
            text = torch.stack(
                [
                    F.pad(ref, (0, text_length_max - ref.shape[1]), value=ignore_id)
                    for ref in text_ref
                ],
                dim=2,
            )

        if text_ref_lengths[0] is not None:
            assert all(ref_lengths.dim() == 1 for ref_lengths in text_ref_lengths), (
                ref_lengths.shape for ref_lengths in text_ref_lengths
            )

        if speech_lengths is not None and text_ref_lengths[0] is not None:
            # Check that batch_size is unified
            assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_ref_lengths[0].shape[0]
            ), (
                speech.shape,
                speech_lengths.shape,
                text.shape,
                text_ref_lengths[0].shape,
            )
        else:
            assert speech.shape[0] == text.shape[0], (speech.shape, text.shape)

        # additional checks with valid src_text
        if "src_text" in kwargs:
            src_text = kwargs["src_text"]
            src_text_lengths = kwargs["src_text_lengths"]

            if src_text is not None:
                assert src_text_lengths.dim() == 1, src_text_lengths.shape
                assert (
                    text_ref[0].shape[0]
                    == src_text.shape[0]
                    == src_text_lengths.shape[0]
                ), (
                    text_ref[0].shape,
                    src_text.shape,
                    src_text_lengths.shape,
                )
        else:
            src_text = None
            src_text_lengths = None

        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        # number of speakers
        # Take the number of speakers from text
        # (= spk_label [Batch, length, num_spk] ) if it is 3-D.
        # This is to handle flexible number of speakers.
        # Used only in "enh + diar" task for now.
        num_spk = text.shape[2] if text.dim() == 3 else self.enh_model.num_spk
        if self.enh_model.num_spk is not None:
            # for compatibility with TCNSeparatorNomask in enh_diar
            assert num_spk == self.enh_model.num_spk, (num_spk, self.enh_model.num_spk)

        # clean speech signal of each speaker
        speech_ref = None
        if self.calc_enh_loss:
            assert "speech_ref1" in kwargs
            speech_ref = [
                kwargs["speech_ref{}".format(spk + 1)] for spk in range(num_spk)
            ]
            # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
            speech_ref = torch.stack(speech_ref, dim=1)
            # for data-parallel
            speech_ref = speech_ref[..., : speech_lengths.max()]
            speech_ref = speech_ref.unbind(dim=1)

        # Calculating enhancement loss
        utt_id = kwargs.get("utt_id", None)
        bypass_enh_flag, skip_enhloss_flag = False, False
        if utt_id is not None and not isinstance(
            self.s2t_model, ESPnetDiarizationModel
        ):
            # TODO(xkc): to pass category info and use predefined category list
            if utt_id[0].endswith("CLEAN"):
                # For clean data
                # feed it to Enhancement, without calculating loss_enh
                bypass_enh_flag = True
                skip_enhloss_flag = True
            elif utt_id[0].endswith("REAL"):
                # For single-speaker real data
                # feed it to Enhancement but without calculating loss_enh
                bypass_enh_flag = False
                skip_enhloss_flag = True
            else:
                # For simulated single-/multi-speaker data
                # feed it to Enhancement and calculate loss_enh
                bypass_enh_flag = False
                skip_enhloss_flag = False

        if not self.calc_enh_loss:
            skip_enhloss_flag = True

        # Bypass the enhancement module
        if (
            self.training and skip_enhloss_flag and not bypass_enh_flag
        ):  # For single-speaker real data: possibility to bypass frontend
            if random.random() <= self.bypass_enh_prob:
                bypass_enh_flag = True

        # 1. Enhancement
        # model forward
        loss_enh = None
        perm = None
        if not bypass_enh_flag:
            ret = self.enh_model.forward_enhance(
                speech, speech_lengths, {"num_spk": num_spk}
            )
            speech_pre, feature_mix, feature_pre, others = ret
            # loss computation
            if not skip_enhloss_flag:
                loss_enh, _, _, perm = self.enh_model.forward_loss(
                    speech_pre,
                    speech_lengths,
                    feature_mix,
                    feature_pre,
                    others,
                    speech_ref,
                )
                loss_enh = loss_enh[0]

                # resort the prediction audios with the obtained permutation
                if perm is not None:
                    speech_pre = ESPnetEnhancementModel.sort_by_perm(speech_pre, perm)
        else:
            speech_pre = [speech]

        # for data-parallel
        if text_ref_lengths[0] is not None:
            text = text[:, :text_length_max]
        if src_text is not None:
            src_text = src_text[:, : src_text_lengths.max()]

        # 2. ASR or ST
        if isinstance(self.s2t_model, ESPnetASRModel):  # ASR
            if perm is None:
                loss_s2t, stats, weight = self.asr_pit_loss(
                    speech_pre, speech_lengths, text.unbind(2), text_ref_lengths
                )
            else:
                loss_s2t, stats, weight = self.s2t_model(
                    torch.cat(speech_pre, dim=0),
                    speech_lengths.repeat(len(speech_pre)),
                    torch.cat(text.unbind(2), dim=0),
                    torch.cat(text_ref_lengths, dim=0),
                )
            stats["loss_asr"] = loss_s2t.detach()
        elif isinstance(self.s2t_model, ESPnetSTModel):  # ST
            loss_s2t, stats, weight = self.s2t_model(
                speech_pre[0],
                speech_lengths,
                text,
                text_ref_lengths[0],
                src_text,
                src_text_lengths,
            )
            stats["loss_st"] = loss_s2t.detach()
        elif isinstance(self.s2t_model, ESPnetDiarizationModel):  # DIAR
            loss_s2t, stats, weight = self.s2t_model(
                speech=speech.clone(),
                speech_lengths=speech_lengths,
                spk_labels=text,
                spk_labels_lengths=text_ref_lengths[0],
                bottleneck_feats=others.get("bottleneck_feats"),
                bottleneck_feats_lengths=others.get("bottleneck_feats_lengths"),
            )
            stats["loss_diar"] = loss_s2t.detach()
        else:
            raise NotImplementedError(f"{type(self.s2t_model)} is not supported yet.")

        if loss_enh is not None:
            loss = loss_enh + loss_s2t
        else:
            loss = loss_s2t

        stats["loss"] = loss.detach() if loss is not None else None
        stats["loss_enh"] = loss_enh.detach() if loss_enh is not None else None

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
            Collect features from the input speech tensor and corresponding lengths.

        This method extracts features using the speech-to-text model, depending on
        whether the model is configured to extract features or generate dummy stats.

        Attributes:
            extract_feats_in_collect_stats (bool): Flag indicating whether to extract
                features in the collection process.

        Args:
            speech (torch.Tensor): The input speech tensor of shape (Batch, Length, ...).
            speech_lengths (torch.Tensor): The lengths of the input speech of shape
                (Batch,).
            **kwargs: Additional keyword arguments. Can include:
                - text (torch.Tensor): The reference text tensor.
                - text_lengths (torch.Tensor): The lengths of the reference text.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - feats (torch.Tensor): The extracted features.
                - feats_lengths (torch.Tensor): The lengths of the extracted features.

        Raises:
            ValueError: If the model configuration does not support feature extraction.

        Examples:
            >>> model = ESPnetEnhS2TModel(...)
            >>> speech = torch.randn(10, 16000)  # Example speech input
            >>> speech_lengths = torch.tensor([16000] * 10)  # Lengths for each input
            >>> feats = model.collect_feats(speech, speech_lengths)
            >>> print(feats["feats"].shape)  # Should print the shape of the extracted features
        """
        if "text" in kwargs:
            text = kwargs["text"]
            text_lengths = kwargs.get("text_lengths", None)
        else:
            text = kwargs["text_spk1"]
            text_lengths = kwargs.get("text_spk1_lengths", None)

        if self.extract_feats_in_collect_stats:
            ret = self.s2t_model.collect_feats(
                speech,
                speech_lengths,
                text,
                text_lengths,
                **kwargs,
            )
            feats, feats_lengths = ret["feats"], ret["feats_lengths"]
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Frontend + Encoder. Note that this method is used by asr_inference.py.

        This method processes the input speech through the enhancement model
        and then encodes the enhanced speech using the speech-to-text model.
        It returns the encoded outputs and their corresponding lengths.

        Args:
            speech: Tensor of shape (Batch, Length, ...), representing the
                input speech signals.
            speech_lengths: Tensor of shape (Batch,), representing the lengths
                of the input speech sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - encoder_out: Encoded output from the speech-to-text model,
                  with shape (Batch, Length, Dim).
                - encoder_out_lens: Tensor of shape (Batch,), representing
                  the lengths of the encoded outputs.

        Raises:
            AssertionError: If the number of speakers in the processed speech
            does not match the expected number of speakers in the enhancement model.

        Examples:
            >>> model = ESPnetEnhS2TModel(...)
            >>> speech = torch.randn(8, 16000)  # 8 samples of 1 second audio
            >>> speech_lengths = torch.tensor([16000] * 8)  # lengths for each sample
            >>> encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
            >>> print(encoder_out.shape)  # Should output: (8, Length, Dim)
            >>> print(encoder_out_lens.shape)  # Should output: (8,)
        """
        (
            speech_pre,
            feature_mix,
            feature_pre,
            others,
        ) = self.enh_model.forward_enhance(speech, speech_lengths)
        num_spk = len(speech_pre)
        assert num_spk == self.enh_model.num_spk, (num_spk, self.enh_model.num_spk)

        encoder_out, encoder_out_lens = zip(
            *[self.s2t_model.encode(sp, speech_lengths) for sp in speech_pre]
        )

        return encoder_out, encoder_out_lens

    def encode_diar(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, num_spk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Frontend + Encoder. Note that this method is used by diar_inference.py.

        This method processes the input speech tensor through the enhancement
        model and then encodes the enhanced speech using the speech-to-text model.
        It is specifically designed for diarization tasks, which involve identifying
        and segmenting speakers in an audio stream.

        Args:
            speech: A tensor of shape (Batch, Length, ...) representing the input
                speech signal.
            speech_lengths: A tensor of shape (Batch,) representing the lengths of
                each speech sample in the batch.
            num_spk: An integer indicating the number of speakers in the input
                speech.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - encoder_out: The output of the encoder after processing the
                    enhanced speech.
                - encoder_out_lens: The lengths of the output sequences from the
                    encoder.
                - speech_pre: The enhanced speech signals.

        Examples:
            >>> model = ESPnetEnhS2TModel(...)
            >>> speech_tensor = torch.randn(2, 16000)  # Batch of 2, 1 second of audio
            >>> speech_lengths = torch.tensor([16000, 16000])  # Lengths of audio
            >>> num_spk = 2  # Assuming there are 2 speakers
            >>> encoder_out, encoder_out_lens, speech_pre = model.encode_diar(
            ...     speech_tensor, speech_lengths, num_spk)

        Note:
            Ensure that the `speech` tensor is pre-processed and in the correct
            format before calling this method. The `num_spk` parameter must match
            the expected number of speakers for accurate processing.

        Raises:
            ValueError: If the input tensor dimensions do not match the expected
                shapes or if `num_spk` is inconsistent with the enhancement model.
        """
        (
            speech_pre,
            _,
            _,
            others,
        ) = self.enh_model.forward_enhance(speech, speech_lengths, {"num_spk": num_spk})
        encoder_out, encoder_out_lens = self.s2t_model.encode(
            speech,
            speech_lengths,
            others.get("bottleneck_feats"),
            others.get("bottleneck_feats_lengths"),
        )

        return encoder_out, encoder_out_lens, speech_pre

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log likelihood (NLL) from transformer decoder.

        This function is typically called within the batchify_nll method. It
        computes the negative log likelihood for a given batch of encoded
        outputs and their corresponding target sequences.

        Args:
            encoder_out: A tensor of shape (Batch, Length, Dim) representing the
                output from the encoder.
            encoder_out_lens: A tensor of shape (Batch,) representing the lengths
                of the encoder outputs.
            ys_pad: A tensor of shape (Batch, Length) representing the padded
                target sequences.
            ys_pad_lens: A tensor of shape (Batch,) representing the lengths of
                the target sequences.

        Returns:
            A tensor representing the computed negative log likelihood for the
            provided inputs.

        Examples:
            >>> encoder_out = torch.randn(32, 50, 256)  # Batch of 32, 50 time steps, 256 features
            >>> encoder_out_lens = torch.randint(1, 50, (32,))  # Random lengths
            >>> ys_pad = torch.randint(0, 100, (32, 40))  # Random target sequences
            >>> ys_pad_lens = torch.randint(1, 40, (32,))  # Random lengths
            >>> nll_value = model.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        """
        return self.s2t_model.nll(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens,
        )

    batchify_nll = ESPnetASRModel.batchify_nll

    def asr_pit_loss(self, speech, speech_lengths, text, text_lengths):
        """
            Calculate the permutation-invariant training (PIT) loss for ASR.

        This method computes the loss for automatic speech recognition (ASR)
        using the permutation-invariant training approach. It determines the
        optimal alignment between the reference and hypothesis sequences
        based on the calculated CTC (Connectionist Temporal Classification)
        loss. The function also sorts the speech input according to the
        optimal permutation.

        Args:
            speech (torch.Tensor): The enhanced speech signals of shape
                (Batch, Length, ...).
            speech_lengths (torch.Tensor): The lengths of the speech signals
                of shape (Batch,).
            text (List[torch.Tensor]): A list of reference text sequences
                for each speaker of shape (Batch, Length).
            text_lengths (List[torch.Tensor]): A list of lengths of the
                reference text sequences for each speaker of shape (Batch,).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
            A tuple containing:
                - loss (torch.Tensor): The calculated loss for the ASR task.
                - stats (Dict[str, torch.Tensor]): A dictionary of statistics
                  related to the loss computation.
                - weight (torch.Tensor): The weight tensor for the computed loss.

        Raises:
            ValueError: If CTC is not used for determining the permutation.

        Examples:
            >>> speech = torch.randn(2, 100, 80)  # Batch of 2, 100 time steps, 80 features
            >>> speech_lengths = torch.tensor([100, 90])
            >>> text = [torch.randint(0, 100, (2, 20)), torch.randint(0, 100, (2, 25))]
            >>> text_lengths = [torch.tensor([20, 25]), torch.tensor([20, 25])]
            >>> loss, stats, weight = model.asr_pit_loss(speech, speech_lengths, text, text_lengths)

        Note:
            Ensure that the `self.s2t_model.ctc` is initialized before calling
            this method, as it is required for the computation of the loss.

        Todo:
            - Implement additional logging for debugging.
        """
        if self.s2t_model.ctc is None:
            raise ValueError("CTC must be used to determine the permutation")
        with torch.no_grad():
            # (B, n_ref, n_hyp)
            loss0 = torch.stack(
                [
                    torch.stack(
                        [
                            self.s2t_model._calc_batch_ctc_loss(
                                speech[h],
                                speech_lengths,
                                text[r],
                                text_lengths[r],
                            )
                            for r in range(self.enh_model.num_spk)
                        ],
                        dim=1,
                    )
                    for h in range(self.enh_model.num_spk)
                ],
                dim=2,
            )
            perm_detail, min_loss = self.permutation_invariant_training(loss0)

        speech = ESPnetEnhancementModel.sort_by_perm(speech, perm_detail)
        loss, stats, weight = self.s2t_model(
            torch.cat(speech, dim=0),
            speech_lengths.repeat(len(speech)),
            torch.cat(text, dim=0),
            torch.cat(text_lengths, dim=0),
        )
        return loss, stats, weight

    def _permutation_loss(self, ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            loss: torch.Tensor: (batch)
            perm: list[(num_spk)]
        """
        num_spk = len(ref)

        losses = torch.stack(
            [
                torch.stack([criterion(ref[r], inf[h]) for r in range(num_spk)], dim=1)
                for h in range(num_spk)
            ],
            dim=2,
        )  # (B, n_ref, n_hyp)
        perm_detail, min_loss = self.permutation_invariant_training(losses)

        return min_loss.mean(), perm_detail

    def permutation_invariant_training(self, losses: torch.Tensor):
        """
        Compute the Permutation Invariant Training (PIT) loss.

        This method applies the Hungarian algorithm to determine the optimal
        assignment of hypotheses to references based on the provided loss
        matrix. The goal is to minimize the total loss by finding the best
        permutation of the hypotheses for each batch.

        Args:
            losses (torch.Tensor): A tensor of shape (batch, nref, nhyp)
                representing the loss values for each reference and hypothesis
                pair.

        Returns:
            perm: list: A list containing the optimal permutation indices for
                each batch, where each entry is of shape (n_spk).
            loss: torch.Tensor: A tensor of shape (batch) representing the
                minimized loss for each batch after applying the optimal
                permutation.

        Examples:
            >>> losses = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            >>> perm, loss = model.permutation_invariant_training(losses)
            >>> print(perm)
            [[0], [1]]
            >>> print(loss)
            tensor([0.1, 0.4])

        Note:
            This method is primarily used in scenarios where multiple
            hypotheses can correspond to multiple references, such as in
            speech recognition or multi-speaker scenarios.

        Raises:
            ValueError: If the cost matrix is infeasible, which can happen
                when all loss values are infinite. In such cases, a random
                assignment will be used.
        """
        hyp_perm, min_perm_loss = [], []
        losses_cpu = losses.data.cpu()
        for b, b_loss in enumerate(losses_cpu):
            # hungarian algorithm
            try:
                row_ind, col_ind = linear_sum_assignment(b_loss)
            except ValueError as err:
                if str(err) == "cost matrix is infeasible":
                    # random assignment since the cost is always inf
                    col_ind = np.array([0, 1])
                    min_perm_loss.append(torch.mean(losses[b, col_ind, col_ind]))
                    hyp_perm.append(col_ind)
                    continue
                else:
                    raise

            min_perm_loss.append(torch.mean(losses[b, row_ind, col_ind]))
            hyp_perm.append(
                torch.as_tensor(col_ind, dtype=torch.long, device=losses.device)
            )

        return hyp_perm, torch.stack(min_perm_loss)

    @typechecked
    def inherite_attributes(
        self,
        inherite_enh_attrs: List[str] = [],
        inherite_s2t_attrs: List[str] = [],
    ):
        """
        Inherit attributes from the enhancement and speech-to-text models.

        This method allows the user to inherit specified attributes from the
        enhancement model and the speech-to-text model, enabling the joint
        model to access properties and methods defined in the respective
        models without directly exposing them.

        Args:
            inherite_enh_attrs (List[str]): A list of attribute names to
                inherit from the enhancement model.
            inherite_s2t_attrs (List[str]): A list of attribute names to
                inherit from the speech-to-text model.

        Examples:
            >>> model = ESPnetEnhS2TModel(enh_model, s2t_model)
            >>> model.inherite_attributes(
            ...     inherite_enh_attrs=['some_enh_attr'],
            ...     inherite_s2t_attrs=['some_s2t_attr']
            ... )
            >>> print(model.some_enh_attr)  # Access inherited attribute
            >>> print(model.some_s2t_attr)  # Access inherited attribute

        Note:
            If the specified attributes do not exist in the respective
            models, their values will be set to None.

        Todo:
            - Consider adding validation to ensure the attributes exist
              in the respective models before setting them.
        """

        if len(inherite_enh_attrs) > 0:
            for attr in inherite_enh_attrs:
                setattr(self, attr, getattr(self.enh_model, attr, None))
        if len(inherite_s2t_attrs) > 0:
            for attr in inherite_s2t_attrs:
                setattr(self, attr, getattr(self.s2t_model, attr, None))
