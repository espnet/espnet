"""Enhancement model module."""

from typing import Dict, Tuple

import torch
from typeguard import typechecked

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.extractor.abs_extractor import AbsExtractor  # noqa
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss  # noqa
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss  # noqa
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper  # noqa
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel  # noqa

EPS = torch.finfo(torch.get_default_dtype()).eps


class ESPnetDiffusionModel(ESPnetEnhancementModel):
    """
    Target Speaker Extraction Frontend model.

    This model implements a frontend for target speaker extraction, which is 
    designed to enhance audio signals by extracting a specific speaker's voice 
    from a mixture of sounds. It combines an encoder, a diffusion process, 
    and a decoder to achieve the desired enhancement.

    Attributes:
        encoder (AbsEncoder): The encoder module for processing audio input.
        diffusion (AbsDiffusion): The diffusion module for signal enhancement.
        decoder (AbsDecoder): The decoder module for reconstructing the output.
        num_spk (int): The number of speakers to enhance (default is 1).
        normalize (bool): A flag indicating whether to normalize the input signals.

    Args:
        encoder (AbsEncoder): The encoder instance to be used.
        diffusion (AbsDiffusion): The diffusion instance for the enhancement.
        decoder (AbsDecoder): The decoder instance to be used.
        num_spk (int): Number of speakers (default is 1).
        normalize (bool): Flag to indicate normalization of input (default is False).
        **kwargs: Additional keyword arguments for the parent class.

    Raises:
        AssertionError: If num_spk is not equal to 1, as only enhancement models 
                        are currently supported.

    Examples:
        >>> model = ESPnetDiffusionModel(encoder, diffusion, decoder, num_spk=1)
        >>> loss, stats, weight = model.forward(speech_mix, speech_mix_lengths, 
        ...                                      speech_ref1=speech_ref1)

    Note:
        - This model is currently limited to enhancement tasks with a single 
          target speaker.
        - The input to the `forward` method requires at least one reference 
          speech signal.

    Todo:
        - Extend the model to support separation tasks for multiple speakers.
    """

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        diffusion: AbsDiffusion,
        decoder: AbsDecoder,
        # loss_wrappers: List[AbsLossWrapper],
        num_spk: int = 1,
        normalize: bool = False,
        **kwargs,
    ):

        super().__init__(
            encoder=encoder,
            separator=None,
            decoder=decoder,
            mask_module=None,
            loss_wrappers=None,
            **kwargs,
        )

        self.encoder = encoder
        self.diffusion = diffusion
        self.decoder = decoder

        # TODO(gituser): Extending the model to separation tasks.
        assert (
            num_spk == 1
        ), "only enhancement models are supported now, num_spk must be 1"
        self.num_spk = num_spk
        self.normalize = normalize

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Frontend + Encoder + Decoder + Calculate loss.

        This method processes the mixed speech input through the encoder and 
        decoder to compute the loss and other statistics. It also handles 
        normalization of the input signals if specified.

        Args:
            speech_mix: A tensor of shape (Batch, samples) or 
                        (Batch, samples, channels) representing the mixed speech.
            speech_mix_lengths: A tensor of shape (Batch,) indicating the lengths 
                               of the mixed speech signals. Default is None, which 
                               is suitable for chunk iterators that do not return 
                               the speech lengths. Refer to 
                               espnet2/iterators/chunk_iter_factory.py for details.
            kwargs: Additional keyword arguments. It must include at least 
                    "speech_ref1" for the first reference signal. Other 
                    reference signals can be provided as "speech_ref2", 
                    etc. Enrollment references can also be passed as 
                    "enroll_ref1", "enroll_ref2", etc.

        Raises:
            AssertionError: If "speech_ref1" is not provided in kwargs, or if the 
                            dimensions of input tensors do not match.

        Returns:
            A tuple containing:
                - loss: A tensor representing the computed loss.
                - stats: A dictionary containing various statistics related to 
                         the forward pass.
                - weight: A tensor representing the weight for the computed loss.

        Examples:
            >>> model = ESPnetDiffusionModel(...)
            >>> speech_mix = torch.randn(2, 16000)  # Batch of 2, 1 second audio
            >>> speech_ref1 = torch.randn(2, 16000)  # Reference for speaker 1
            >>> loss, stats, weight = model.forward(
            ...     speech_mix,
            ...     speech_ref1=speech_ref1,
            ...     speech_mix_lengths=torch.tensor([16000, 16000])
            ... )

        Note:
            The method assumes that the batch size of `speech_mix` and 
            `speech_ref` tensors are the same.
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
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)
        batch_size = speech_mix.shape[0]
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
        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)
        speech_mix = speech_mix[:, : speech_lengths.max()]

        if self.normalize:
            normfac = speech_mix.abs().max() * 1.1 + 1e-5
        else:
            normfac = 1.0

        speech_mix = speech_mix / normfac
        speech_ref = [r / normfac for r in speech_ref]

        # loss computation
        loss, stats, weight = self.forward_loss(
            speech_ref=speech_ref, speech_mix=speech_mix, speech_lengths=speech_lengths
        )
        return loss, stats, weight

    def enhance(self, feature_mix):
        """
        Enhancement model module for speaker extraction using diffusion processes.

    This module defines the ESPnetDiffusionModel class, which is designed for
    target speaker extraction through a combination of an encoder, a diffusion
    process, and a decoder. It normalizes input features if specified and
    computes loss during the forward pass.

    Attributes:
        encoder (AbsEncoder): The encoder component of the model.
        diffusion (AbsDiffusion): The diffusion process component of the model.
        decoder (AbsDecoder): The decoder component of the model.
        num_spk (int): The number of speakers (default is 1).
        normalize (bool): Flag to indicate whether to normalize the input
            features (default is False).

    Args:
        encoder (AbsEncoder): The encoder instance to process the input.
        diffusion (AbsDiffusion): The diffusion instance for enhancing features.
        decoder (AbsDecoder): The decoder instance to reconstruct audio signals.
        num_spk (int, optional): Number of speakers (default is 1).
        normalize (bool, optional): Flag for normalizing input features
            (default is False).
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
        containing the computed loss, statistics, and weight for backpropagation.

    Raises:
        AssertionError: If `num_spk` is not equal to 1, as only enhancement
        models are currently supported.

    Examples:
        # Create model components
        encoder = MyEncoder(...)
        diffusion = MyDiffusion(...)
        decoder = MyDecoder(...)

        # Instantiate the model
        model = ESPnetDiffusionModel(encoder, diffusion, decoder, num_spk=1)

        # Forward pass with mixed speech and reference signals
        loss, stats, weight = model.forward(speech_mix, speech_mix_lengths,
                                            speech_ref1=speech_ref1)

        # Enhance features
        enhanced_features = model.enhance(feature_mix)

    Note:
        The model currently only supports single-speaker enhancement tasks.
        """
        if self.normalize:
            normfac = feature_mix.abs().max() * 1.1 + 1e-5
            feature_mix = feature_mix / normfac

        return self.diffusion.enhance(feature_mix)

    def forward_loss(
        self,
        speech_ref,
        speech_mix,
        speech_lengths,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute the forward loss for the speech enhancement model.

    This method processes the mixed speech and reference signals through
    the encoder, computes the diffusion-based loss, and returns the loss
    along with relevant statistics and weight for the batch.

    Args:
        speech_ref (List[torch.Tensor]): A list of reference speech signals
            for each speaker. Each tensor should be of shape (Batch, samples).
        speech_mix (torch.Tensor): The mixed speech signal of shape
            (Batch, samples) or (Batch, samples, channels).
        speech_lengths (torch.Tensor): A tensor of shape (Batch,) containing
            the lengths of the mixed speech signals.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
        containing:
            - loss (torch.Tensor): The computed loss for the batch.
            - stats (Dict[str, torch.Tensor]): A dictionary containing
              statistics related to the loss, such as 'loss'.
            - weight (torch.Tensor): The weight tensor for the batch.

    Examples:
        >>> model = ESPnetDiffusionModel(...)
        >>> speech_mix = torch.randn(8, 16000)  # Example mixed speech
        >>> speech_ref = [torch.randn(8, 16000)]  # Example reference
        >>> speech_lengths = torch.tensor([16000] * 8)  # Lengths
        >>> loss, stats, weight = model.forward_loss(speech_ref, speech_mix, speech_lengths)

    Note:
        Ensure that the input reference signals are correctly shaped
        and that at least one reference signal is provided.

    Raises:
        AssertionError: If the dimensions of the input tensors do not match
        or if no reference signal is provided.
        """
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        feature_ref, flens = self.encoder(speech_ref[0], speech_lengths)

        stats = {}
        loss = self.diffusion(feature_ref=feature_ref, feature_mix=feature_mix)
        stats["loss"] = loss.detach()
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Collect features from the mixed speech input.

        This method processes the mixed speech input tensor and its 
        corresponding lengths to return a dictionary containing the 
        features and their lengths. It is typically used in data-parallel 
        scenarios to prepare input for further processing.

        Args:
            speech_mix: A tensor of shape (Batch, samples) or 
                        (Batch, samples, channels) representing the 
                        mixed speech signals.
            speech_mix_lengths: A tensor of shape (Batch,) containing the 
                               lengths of each mixed speech signal.
            kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing:
                - "feats": The processed speech mix tensor.
                - "feats_lengths": The lengths of the processed speech mix.

        Examples:
            >>> speech_mix = torch.randn(4, 16000)  # 4 samples, 16000 time steps
            >>> speech_mix_lengths = torch.tensor([16000, 16000, 16000, 16000])
            >>> model = ESPnetDiffusionModel(...)
            >>> features = model.collect_feats(speech_mix, speech_mix_lengths)
            >>> print(features["feats"].shape)  # Output: torch.Size([4, 16000])
            >>> print(features["feats_lengths"])  # Output: tensor([16000, 16000, 16000, 16000])
        """
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
