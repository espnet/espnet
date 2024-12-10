"""Enhancement model module."""

import contextlib
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
from typeguard import typechecked

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetExtractionModel(AbsESPnetModel):
    """
    ESPnetExtractionModel is a target speaker extraction frontend model.

This model integrates an encoder, extractor, and decoder to perform target
speaker extraction from mixed audio signals. It supports multiple loss
wrappers and can handle a flexible number of speakers.

Attributes:
    encoder (AbsEncoder): The encoder used for processing input audio.
    extractor (AbsExtractor): The extractor used to separate the target
        speakers from the mixture.
    decoder (AbsDecoder): The decoder that reconstructs the separated audio.
    loss_wrappers (List[AbsLossWrapper]): A list of loss wrappers for loss
        computation.
    num_spk (int): The number of target speakers to extract (default: 1).
    flexible_numspk (bool): If True, num_spk is regarded as the maximum
        possible number of speakers (default: False).
    share_encoder (bool): Whether to share the encoder for both mixture and
        enrollment (default: True).
    extract_feats_in_collect_stats (bool): If True, features are extracted
        during statistics collection (default: False).
    ref_channel (int): The reference channel for multi-channel signals.

Args:
    encoder (AbsEncoder): The encoder to use for the model.
    extractor (AbsExtractor): The extractor to use for the model.
    decoder (AbsDecoder): The decoder to use for the model.
    loss_wrappers (List[AbsLossWrapper]): List of loss wrappers for training.
    num_spk (int, optional): Number of target speakers (default: 1).
    flexible_numspk (bool, optional): Allow flexible number of speakers
        (default: False).
    share_encoder (bool, optional): Share encoder between mixture and
        enrollment (default: True).
    extract_feats_in_collect_stats (bool, optional): Extract features during
        collect stats (default: False).

Raises:
    ValueError: If there are duplicated loss names or unsupported loss types.

Examples:
    # Example usage:
    encoder = SomeEncoder()
    extractor = SomeExtractor()
    decoder = SomeDecoder()
    loss_wrapper = SomeLossWrapper()
    
    model = ESPnetExtractionModel(
        encoder=encoder,
        extractor=extractor,
        decoder=decoder,
        loss_wrappers=[loss_wrapper],
        num_spk=2,
        flexible_numspk=True
    )

    # Forward pass
    speech_mix = torch.randn(4, 16000)  # (Batch, Samples)
    speech_lengths = torch.tensor([16000] * 4)  # Lengths for each batch
    speech_ref1 = torch.randn(4, 16000)  # Reference for speaker 1
    speech_ref2 = torch.randn(4, 16000)  # Reference for speaker 2

    loss, stats, weight = model.forward(
        speech_mix,
        speech_lengths=speech_lengths,
        speech_ref1=speech_ref1,
        speech_ref2=speech_ref2,
        enroll_ref1=speech_ref1,
        enroll_ref2=speech_ref2
    )
    """

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        extractor: AbsExtractor,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        num_spk: int = 1,
        flexible_numspk: bool = False,
        share_encoder: bool = True,
        extract_feats_in_collect_stats: bool = False,
    ):

        super().__init__()

        self.encoder = encoder
        self.extractor = extractor
        self.decoder = decoder
        # Whether to share encoder for both mixture and enrollment
        self.share_encoder = share_encoder
        self.num_spk = num_spk
        # If True, self.num_spk is regarded as the MAXIMUM possible number of speakers
        self.flexible_numspk = flexible_numspk

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))
        for w in self.loss_wrappers:
            if getattr(w.criterion, "is_noise_loss", False):
                raise ValueError("is_noise_loss=True is not supported")
            elif getattr(w.criterion, "is_dereverb_loss", False):
                raise ValueError("is_dereverb_loss=True is not supported")

        # for multi-channel signal
        self.ref_channel = getattr(self.extractor, "ref_channel", None)
        if self.ref_channel is None:
            self.ref_channel = 0

        # Used in espnet2/tasks/abs_task.py for determining whether or not to do
        # collect_feats during collect stats (stage 5).
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Decoder + Calculate loss.

        This method processes the input speech mixture through the frontend,
        encoder, and decoder, and computes the loss based on the reference
        speech signals and enrollment references provided in `kwargs`.

        Args:
            speech_mix: A tensor of shape (Batch, samples) or
                (Batch, samples, channels) representing the mixed speech
                signals.
            speech_mix_lengths: A tensor of shape (Batch,) representing the
                lengths of the mixed speech signals. Defaults to None,
                which is used for chunk iterators that do not return
                speech lengths (see espnet2/iterators/chunk_iter_factory.py).
            kwargs: Additional keyword arguments. It must include:
                - "speech_ref1": (Batch, samples) or
                  (Batch, samples, channels) for the reference signal of
                  speaker 1.
                - "enroll_ref1": (Batch, samples_aux) for enrollment (raw
                  audio or embedding) for speaker 1.
                - Additional enrollment references can be included as
                  "speech_ref2", "enroll_ref2", etc.

        Returns:
            A tuple containing:
                - loss: A tensor representing the computed loss.
                - stats: A dictionary containing various statistics.
                - weight: A tensor representing the weight for the loss.

        Raises:
            AssertionError: If the required reference signals are not
                provided in `kwargs` or if their shapes are inconsistent.

        Examples:
            >>> model = ESPnetExtractionModel(...)
            >>> speech_mix = torch.randn(4, 16000)  # Example mixed speech
            >>> speech_ref1 = torch.randn(4, 16000)  # Example reference for speaker 1
            >>> enroll_ref1 = torch.randn(4, 8000)   # Example enrollment for speaker 1
            >>> loss, stats, weight = model.forward(
            ...     speech_mix,
            ...     speech_ref1=speech_ref1,
            ...     enroll_ref1=enroll_ref1
            ... )

        Note:
            The method expects at least one reference signal and one
            enrollment signal to be provided in `kwargs`. The number of
            reference signals must match the number of speakers defined in
            the model.
        """
        # reference speech signal of each speaker
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}",
                torch.zeros_like(kwargs["speech_ref1"]),
            )
            for spk in range(self.num_spk)
            if "speech_ref{}".format(spk + 1) in kwargs
        ]
        num_spk = len(speech_ref) if self.flexible_numspk else self.num_spk
        assert len(speech_ref) == num_spk, (len(speech_ref), num_spk)
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)
        batch_size = speech_mix.shape[0]

        assert "enroll_ref1" in kwargs, "At least 1 enrollment signal is required."
        # enrollment signal for each speaker (as the target)
        enroll_ref = [
            # (Batch, samples_aux)
            kwargs["enroll_ref{}".format(spk + 1)]
            for spk in range(num_spk)
            if "enroll_ref{}".format(spk + 1) in kwargs
        ]
        enroll_ref_lengths = [
            # (Batch,)
            kwargs.get(
                "enroll_ref{}_lengths".format(spk + 1),
                torch.ones(batch_size).int().fill_(enroll_ref[spk].size(1)),
            )
            for spk in range(num_spk)
            if "enroll_ref{}".format(spk + 1) in kwargs
        ]

        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        for aux in enroll_ref:
            assert aux.shape[0] == speech_mix.shape[0], (aux.shape, speech_mix.shape)

        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)

        speech_mix = speech_mix[:, : speech_lengths.max()]
        enroll_ref = [
            enroll_ref[spk][:, : enroll_ref_lengths[spk].max()]
            for spk in range(len(enroll_ref))
        ]
        assert len(speech_ref) == len(enroll_ref), (len(speech_ref), len(enroll_ref))

        additional = {}
        # Additional data for training the TSE model
        if self.flexible_numspk:
            additional["num_spk"] = num_spk

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, enroll_ref, enroll_ref_lengths, additional
        )

        # loss computation
        loss, stats, weight, perm = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
        )
        return loss, stats, weight

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        enroll_ref: torch.Tensor,
        enroll_ref_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhances the input mixed speech signal using the encoder and extractor.

        This method processes the mixed speech input and reference signals for 
        enrollment to produce enhanced speech outputs. It uses the encoder to 
        extract features from the mixed speech and reference signals, which are 
        then processed by the extractor to generate the enhanced signals.

        Args:
            speech_mix: Tensor of shape (Batch, samples) or 
                        (Batch, samples, channels) representing the mixed speech.
            speech_lengths: Tensor of shape (Batch,) indicating the lengths of 
                            the mixed speech signals.
            enroll_ref: Tensor of shape (Batch, samples_aux) or 
                        (Batch, samples_aux, channels) representing the enrollment 
                        reference signals for each speaker.
            enroll_ref_lengths: Tensor of shape (Batch,) indicating the lengths 
                                of the enrollment reference signals.
            additional: Optional dictionary containing additional parameters 
                        for enhancement. Default is None.

        Returns:
            Tuple containing:
                - speech_pre: Enhanced speech tensor of shape (Batch, samples) 
                              or (Batch, samples, channels).
                - feature_mix: Features extracted from the mixed speech.
                - feature_pre: Features extracted from the enhanced speech.

        Examples:
            >>> model = ESPnetExtractionModel(...)
            >>> enhanced_speech, features_mix, features_pre = model.forward_enhance(
            ...     speech_mix, speech_lengths, enroll_ref, enroll_ref_lengths
            ... )

        Note:
            This method is designed to work with both single and multi-channel 
            signals. The extraction of features and enhancement is based on the 
            provided enrollment references for the target speakers.

        Raises:
            ValueError: If the input dimensions do not match the expected shapes 
                        or if the reference signals are not provided as required.
        """
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        if self.share_encoder:
            feature_aux, flens_aux = zip(
                *[
                    self.encoder(enroll_ref[spk], enroll_ref_lengths[spk])
                    for spk in range(len(enroll_ref))
                ]
            )
        else:
            feature_aux = enroll_ref
            flens_aux = enroll_ref_lengths

        feature_pre, _, others = zip(
            *[
                self.extractor(
                    feature_mix,
                    flens,
                    feature_aux[spk],
                    flens_aux[spk],
                    suffix_tag=f"_spk{spk + 1}",
                    additional=additional,
                )
                for spk in range(len(enroll_ref))
            ]
        )
        others = {k: v for dic in others for k, v in dic.items()}
        if feature_pre[0] is not None:
            speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            speech_pre = None
        return speech_pre, feature_mix, feature_pre, others

    def forward_loss(
        self,
        speech_pre: torch.Tensor,
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: torch.Tensor,
        others: OrderedDict,
        speech_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Calculates the forward loss for the target speaker extraction model.

    This method computes the loss between the predicted speech signals and the
    reference speech signals using specified loss criteria. It supports both 
    time-domain and frequency-domain losses. The method aggregates loss values 
    from multiple loss wrappers and returns the overall loss, along with 
    additional statistics.

    Args:
        speech_pre: (Batch, samples) or (Batch, samples, channels) - The 
            predicted speech signals from the model.
        speech_lengths: (Batch,) - A tensor indicating the lengths of the 
            predicted speech signals.
        feature_mix: (Batch, feature_dim, samples) - The mixed speech 
            features extracted from the input mixed signals.
        feature_pre: (Batch, num_speakers, feature_dim, samples) - The 
            features of the predicted speech signals.
        others: OrderedDict - Additional data required for loss computation, 
            such as masks or other auxiliary information.
        speech_ref: (Batch, num_speakers, samples) - The reference speech 
            signals for each target speaker.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: 
            - loss: The computed loss value.
            - stats: A dictionary containing additional statistics from the loss 
            computation.
            - weight: The weight used for the loss computation.

    Raises:
        NotImplementedError: If an unsupported loss type is encountered in 
            the loss wrappers.
        AttributeError: If all criteria have only_for_test=True during 
            training.

    Examples:
        loss, stats, weight, perm = model.forward_loss(
            speech_pre, speech_lengths, feature_mix, feature_pre, others, speech_ref
        )

    Note:
        This method is designed to work with multiple loss wrappers that are 
        defined during model initialization. Ensure that the criteria used are 
        compatible with the input tensors provided.
        """
        loss = 0.0
        stats = {}
        o = {}
        perm = None
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if getattr(criterion, "only_for_test", False) and self.training:
                continue

            zero_weight = loss_wrapper.weight == 0.0
            if isinstance(criterion, TimeDomainLoss):
                assert speech_pre is not None
                sref, spre = self._align_ref_pre_channels(
                    speech_ref, speech_pre, ch_dim=2, force_1ch=True
                )
                # for the time domain criterions
                with torch.no_grad() if zero_weight else contextlib.ExitStack():
                    l, s, o = loss_wrapper(sref, spre, {**others, **o})
            elif isinstance(criterion, FrequencyDomainLoss):
                sref, spre = self._align_ref_pre_channels(
                    speech_ref, speech_pre, ch_dim=2, force_1ch=False
                )
                # for the time-frequency domain criterions
                if criterion.compute_on_mask:
                    # compute loss on masks
                    tf_ref, tf_pre = self._get_speech_masks(
                        criterion,
                        feature_mix,
                        None,
                        speech_ref,
                        speech_pre,
                        speech_lengths,
                        others,
                    )
                else:
                    # compute on spectrum
                    tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in sref]
                    tf_pre = [self.encoder(sp, speech_lengths)[0] for sp in spre]

                with torch.no_grad() if zero_weight else contextlib.ExitStack():
                    l, s, o = loss_wrapper(tf_ref, tf_pre, {**others, **o})
            else:
                raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

            loss += l * loss_wrapper.weight
            stats.update(s)

            if perm is None and "perm" in o:
                perm = o["perm"]

        if self.training and isinstance(loss, float):
            raise AttributeError(
                "At least one criterion must satisfy: only_for_test=False"
            )
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight, perm

    def _align_ref_pre_channels(self, ref, pre, ch_dim=2, force_1ch=False):
        if ref is None or pre is None:
            return ref, pre
        # NOTE: input must be a list of time-domain signals
        index = ref[0].new_tensor(self.ref_channel, dtype=torch.long)

        # for models like SVoice that output multiple lists of separated signals
        pre_is_multi_list = isinstance(pre[0], (list, tuple))
        pre_dim = pre[0][0].dim() if pre_is_multi_list else pre[0].dim()

        if ref[0].dim() > pre_dim:
            # multi-channel reference and single-channel output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
        elif ref[0].dim() < pre_dim:
            # single-channel reference and multi-channel output
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        elif ref[0].dim() == pre_dim == 3 and force_1ch:
            # multi-channel reference and output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        return ref, pre

    def _get_speech_masks(
        self, criterion, feature_mix, noise_ref, speech_ref, speech_pre, ilens, others
    ):
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(sr, ilens)[0] for sr in speech_ref],
            noise_spec=noise_spec,
        )
        if "mask_spk1" in others:
            masks_pre = [
                others["mask_spk{}".format(spk + 1)]
                for spk in range(self.num_spk)
                if "mask_dereverb{}".format(spk + 1) in others
            ]
        else:
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(sp, ilens)[0] for sp in speech_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Collect features from the input speech mixture for data-parallel processing.

    This method extracts features from the input speech mixture tensor and its 
    lengths. It prepares the data for further processing, ensuring that the 
    speech mixture is correctly sized for data-parallel operations.

    Args:
        speech_mix: A tensor of shape (Batch, samples) or 
                    (Batch, samples, channels) representing the input 
                    speech mixture.
        speech_mix_lengths: A tensor of shape (Batch,) representing the lengths 
                        of the input speech mixture for each batch item.
        **kwargs: Additional keyword arguments.

    Returns:
        A dictionary containing:
            - "feats": A tensor containing the extracted features.
            - "feats_lengths": A tensor containing the lengths of the features.

    Examples:
        >>> speech_mix = torch.randn(4, 16000)  # Example with 4 batches of 1s audio
        >>> speech_mix_lengths = torch.tensor([16000, 16000, 16000, 16000])
        >>> model = ESPnetExtractionModel(...)
        >>> features = model.collect_feats(speech_mix, speech_mix_lengths)
        >>> print(features["feats"].shape)  # Should be (4, 16000)
        >>> print(features["feats_lengths"])  # Should be tensor of lengths

    Note:
        This method is particularly useful in scenarios where data-parallel 
        processing is required, ensuring that the features are gathered and 
        aligned correctly.
        """
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
