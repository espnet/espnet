import argparse
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import editdistance
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.uasr.generator.abs_generator import AbsGenerator
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


try:
    import kenlm  # for CI import
except (ImportError, ModuleNotFoundError):
    kenlm = None


class ESPnetUASRModel(AbsESPnetModel):
    """
    Unsupervised ASR model.

    This model is designed for unsupervised automatic speech recognition (ASR) tasks.
    The implementation is based on the work from FAIRSEQ:
    https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/unsupervised

    Attributes:
        frontend (Optional[AbsFrontend]): The frontend module for feature extraction.
        segmenter (Optional[AbsSegmenter]): The segmenter module for processing features.
        generator (AbsGenerator): The generator module for producing output.
        discriminator (AbsDiscriminator): The discriminator module for classification.
        losses (Dict[str, AbsUASRLoss]): A dictionary containing various loss functions.
        kenlm_path (Optional[str]): Path to the KenLM language model.
        token_list (Optional[list]): List of tokens for text representation.
        max_epoch (Optional[int]): Maximum number of training epochs.
        vocab_size (int): Size of the vocabulary.
        cfg (Optional[Dict], optional): Configuration options.
        pad (int): Padding index.
        sil_token (str): Silence token representation.
        sos_token (str): Start-of-sequence token representation.
        eos_token (str): End-of-sequence token representation.
        skip_softmax (str2bool): Whether to skip softmax in generation.
        use_gumbel (str2bool): Whether to use Gumbel softmax.
        use_hard_gumbel (str2bool): Whether to use hard Gumbel softmax.
        min_temperature (float): Minimum temperature for Gumbel softmax.
        max_temperature (float): Maximum temperature for Gumbel softmax.
        decay_temperature (float): Decay factor for temperature.
        use_collected_training_feats (str2bool): Whether to use collected features.

    Args:
        frontend (Optional[AbsFrontend]): The frontend module for feature extraction.
        segmenter (Optional[AbsSegmenter]): The segmenter module for processing features.
        generator (AbsGenerator): The generator module for producing output.
        discriminator (AbsDiscriminator): The discriminator module for classification.
        losses (Dict[str, AbsUASRLoss]): A dictionary containing various loss functions.
        kenlm_path (Optional[str]): Path to the KenLM language model.
        token_list (Optional[list]): List of tokens for text representation.
        max_epoch (Optional[int]): Maximum number of training epochs.
        vocab_size (int): Size of the vocabulary.
        cfg (Optional[Dict], optional): Configuration options.
        pad (int): Padding index (default: 1).
        sil_token (str): Silence token representation (default: "<SIL>").
        sos_token (str): Start-of-sequence token representation (default: "<s>").
        eos_token (str): End-of-sequence token representation (default: "</s>").
        skip_softmax (str2bool): Whether to skip softmax in generation (default: False).
        use_gumbel (str2bool): Whether to use Gumbel softmax (default: False).
        use_hard_gumbel (str2bool): Whether to use hard Gumbel softmax (default: True).
        min_temperature (float): Minimum temperature for Gumbel softmax (default: 0.1).
        max_temperature (float): Maximum temperature for Gumbel softmax (default: 2.0).
        decay_temperature (float): Decay factor for temperature (default: 0.99995).
        use_collected_training_feats (str2bool): Whether to use collected features (default: False).

    Raises:
        AssertionError: If KenLM is not installed or if invalid parameters are provided.

    Examples:
        model = ESPnetUASRModel(
            frontend=my_frontend,
            segmenter=my_segmenter,
            generator=my_generator,
            discriminator=my_discriminator,
            losses=my_losses,
            kenlm_path='path/to/kenlm',
            token_list=my_token_list,
            max_epoch=50,
            vocab_size=1000,
            pad=1,
            sil_token='<SIL>',
            sos_token='<s>',
            eos_token='</s>',
            skip_softmax=False,
            use_gumbel=True,
            use_hard_gumbel=True,
            min_temperature=0.1,
            max_temperature=2.0,
            decay_temperature=0.99995,
            use_collected_training_feats=False,
        )
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        segmenter: Optional[AbsSegmenter],
        generator: AbsGenerator,
        discriminator: AbsDiscriminator,
        losses: Dict[str, AbsUASRLoss],
        kenlm_path: Optional[str],
        token_list: Optional[list],
        max_epoch: Optional[int],
        vocab_size: int,
        cfg: Optional[Dict] = None,
        pad: int = 1,
        sil_token: str = "<SIL>",
        sos_token: str = "<s>",
        eos_token: str = "</s>",
        skip_softmax: str2bool = False,
        use_gumbel: str2bool = False,
        use_hard_gumbel: str2bool = True,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0,
        decay_temperature: float = 0.99995,
        use_collected_training_feats: str2bool = False,
    ):

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.frontend = frontend
        self.segmenter = segmenter
        self.use_segmenter = True if segmenter is not None else False
        self.generator = generator
        self.discriminator = discriminator
        self.pad = pad
        if cfg is not None:
            cfg = argparse.Namespace(**cfg)
            self.skip_softmax = cfg.no_softmax
            self.use_gumbel = cfg.gumbel
            self.use_hard_gumbel = cfg.hard_gumbel
        else:
            self.skip_softmax = skip_softmax
            self.use_gumbel = use_gumbel
            self.use_hard_gumbel = use_hard_gumbel

        self.use_collected_training_feats = use_collected_training_feats

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.decay_temperature = decay_temperature
        self.current_temperature = max_temperature
        self._number_updates = 0
        self._number_epochs = 0

        self.max_epoch = max_epoch
        # for loss registration
        self.losses = torch.nn.ModuleDict(losses)

        # for validation
        self.vocab_size = vocab_size
        self.token_list = token_list
        self.token_id_converter = TokenIDConverter(token_list=token_list)
        self.sil = self.token_id_converter.tokens2ids([sil_token])[0]
        self.sos = self.token_id_converter.tokens2ids([sos_token])[0]
        self.eos = self.token_id_converter.tokens2ids([eos_token])[0]

        self.kenlm = None
        assert (
            kenlm is not None
        ), "kenlm is not installed, please install from tools/installers"
        if kenlm_path:
            self.kenlm = kenlm.Model(kenlm_path)

    @property
    def number_updates(self):
        return self._number_updates

    @number_updates.setter
    @typechecked
    def number_updates(self, iiter: int):
        """
        Get the number of updates for the model.

        This property returns the current number of updates that the model has
        undergone. It is used to keep track of the training progress.

        Returns:
            int: The number of updates.

        Examples:
            >>> model = ESPnetUASRModel(...)
            >>> print(model.number_updates)
            0
        """
        assert iiter >= 0
        self._number_updates = iiter

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
        pseudo_labels_lengths: Optional[torch.Tensor] = None,
        do_validation: Optional[str2bool] = False,
        print_hyp: Optional[str2bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Processes input speech data through the model components.

        The forward method performs the following operations:
        1. Extracts features from the input speech.
        2. Generates fake samples using the generator.
        3. Optionally applies segmentation to the generated samples.
        4. Calculates losses based on discriminator predictions.
        5. If validation is enabled, computes validation statistics.

        Args:
            speech (torch.Tensor): Input speech tensor of shape
                (batch_size, sequence_length).
            speech_lengths (torch.Tensor): Lengths of the input speech
                sequences of shape (batch_size,).
            text (Optional[torch.Tensor]): Ground truth text tensor of shape
                (batch_size, max_text_length). Default is None.
            text_lengths (Optional[torch.Tensor]): Lengths of the text sequences
                of shape (batch_size,). Default is None.
            pseudo_labels (Optional[torch.Tensor]): Pseudo labels for
                training of shape (batch_size, max_label_length). Default is None.
            pseudo_labels_lengths (Optional[torch.Tensor]): Lengths of pseudo
                labels of shape (batch_size,). Default is None.
            do_validation (Optional[str2bool]): Whether to perform validation
                during training. Default is False.
            print_hyp (Optional[str2bool]): Whether to print hypotheses during
                validation. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
            containing:
                - loss (torch.Tensor): The computed loss for the batch.
                - stats (Dict[str, torch.Tensor]): A dictionary containing
                    various statistics from the forward pass.
                - weight (torch.Tensor): The weight for the current batch.

        Raises:
            AssertionError: If the input dimensions do not match expected shapes.

        Examples:
            >>> model = ESPnetUASRModel(...)
            >>> speech = torch.randn(32, 16000)  # 32 samples of 1 second
            >>> speech_lengths = torch.tensor([16000] * 32)  # All samples are 1s
            >>> text = torch.randint(0, 100, (32, 20))  # Random text
            >>> text_lengths = torch.tensor([20] * 32)  # All text lengths are 20
            >>> loss, stats, weight = model.forward(speech, speech_lengths, text,
            ...                                       text_lengths)
        """
        stats = {}

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Feats encode (Extract feats + Apply segmenter)
        feats, padding_mask = self.encode(speech, speech_lengths)

        # 2. Generate fake samples
        (
            generated_sample,
            real_sample,
            x_inter,
            generated_sample_padding_mask,
        ) = self.generator(feats, text, padding_mask)

        # 3. Reprocess segments
        if self.use_segmenter:
            (
                generated_sample,
                generated_sample_padding_mask,
            ) = self.segmenter.logit_segment(
                generated_sample, generated_sample_padding_mask
            )

        # for phone_diversity_loss
        generated_sample_logits = generated_sample

        if not self.skip_softmax:
            if self.training and self.use_gumbel:
                generated_sample = F.gumbel_softmax(
                    generated_sample_logits.float(),
                    tau=self.curr_temp,
                    hard=self.use_hard_gumbel,
                ).type_as(generated_sample_logits)
            else:
                generated_sample = generated_sample_logits.softmax(-1)

        # for validation
        vocab_seen = None
        if do_validation:
            batch_num_errors = 0
            batched_hyp_ids = generated_sample.argmax(-1)
            batched_hyp_ids[generated_sample_padding_mask] = self.pad

            # for kenlm ppl metric
            batch_lm_log_prob = 0
            batch_num_hyp_tokens = 0
            vocab_seen = torch.zeros(self.vocab_size - 4, dtype=torch.bool)

            for hyp_ids, ref_ids in zip(batched_hyp_ids, text):
                # remove <pad> and <unk>
                hyp_ids = hyp_ids[hyp_ids >= 4]
                # remove duplicate tokens
                hyp_ids = hyp_ids.unique_consecutive()
                # remove silence
                hyp_ids_nosil = hyp_ids[hyp_ids != self.sil]
                hyp_ids_nosil_list = hyp_ids_nosil.tolist()

                if self.kenlm:
                    hyp_token_list = self.token_id_converter.ids2tokens(
                        integers=hyp_ids
                    )
                    hyp_tokens = " ".join(hyp_token_list)
                    lm_log_prob = self.kenlm.score(hyp_tokens)
                    batch_lm_log_prob += lm_log_prob
                    batch_num_hyp_tokens += len(hyp_token_list)

                    hyp_tokens_index = hyp_ids[hyp_ids >= 4]
                    vocab_seen[hyp_tokens_index - 4] = True

                ref_ids = ref_ids[ref_ids != self.pad]
                ref_ids_list = ref_ids.tolist()
                num_errors = editdistance.eval(hyp_ids_nosil_list, ref_ids_list)
                batch_num_errors += num_errors

            stats["batch_num_errors"] = batch_num_errors
            stats["batch_num_ref_tokens"] = text_lengths.sum().item()
            if self.kenlm:
                stats["batch_lm_log_prob"] = batch_lm_log_prob
                stats["batch_num_hyp_tokens"] = batch_num_hyp_tokens
                stats["batch_size"] = batch_size

            # print the last sample in the batch
            if print_hyp:
                hyp_token_list = self.token_id_converter.ids2tokens(
                    integers=hyp_ids_nosil
                )
                hyp_tokens = " ".join(hyp_token_list)

                ref_token_list = self.token_id_converter.ids2tokens(integers=ref_ids)
                ref_tokens = " ".join(ref_token_list)
                logging.info(f"[REF]: {ref_tokens}")
                logging.info(f"[HYP]: {hyp_tokens}")

        real_sample_padding_mask = text == self.pad
        # 5. Discriminator condition
        generated_sample_prediction = self.discriminator(
            generated_sample, generated_sample_padding_mask
        )
        real_sample_prediction = self.discriminator(
            real_sample, real_sample_padding_mask
        )

        is_discriminative_step = self.is_discriminative_step()

        # 5. Calculate losses
        loss_info = []

        if "discriminator_loss" in self.losses.keys():
            (
                generated_sample_prediction_loss,
                real_sample_prediction_loss,
            ) = self.losses["discriminator_loss"](
                generated_sample_prediction,
                real_sample_prediction,
                is_discriminative_step,
            )
            loss_info.append(
                generated_sample_prediction_loss
                * self.losses["discriminator_loss"].weight
            )
            if is_discriminative_step:
                loss_info.append(
                    real_sample_prediction_loss
                    * self.losses["discriminator_loss"].weight
                )
        else:
            generated_sample_prediction_loss, real_sample_prediction_loss = None, None

        if "gradient_penalty" in self.losses.keys():
            gp = self.losses["gradient_penalty"](
                generated_sample,
                real_sample,
                self.training,
                is_discriminative_step,
            )
            loss_info.append(gp * self.losses["gradient_penalty"].weight)
            loss_info.append(gp * self.losses["gradient_penalty"].weight)
        else:
            gp = None

        if "phoneme_diversity_loss" in self.losses.keys():
            pdl = self.losses["phoneme_diversity_loss"](
                generated_sample_logits, batch_size, is_discriminative_step
            )
            loss_info.append(pdl * self.losses["phoneme_diversity_loss"].weight)
        else:
            pdl = None

        if "smoothness_penalty" in self.losses.keys():
            sp = self.losses["smoothness_penalty"](
                generated_sample_logits,
                generated_sample_padding_mask,
                batch_size,
                is_discriminative_step,
            )
            loss_info.append(sp * self.losses["smoothness_penalty"].weight)
        else:
            sp = None

        if "pseudo_label_loss" in self.losses.keys() and pseudo_labels is not None:
            mmi = self.losses["pseudo_label_loss"](
                x_inter, pseudo_labels, is_discriminative_step
            )
            loss_info.append(mmi * self.losses["pseudo_label_loss"].weight)
        else:
            mmi = None

        # Update temperature
        self._change_temperature()
        self.number_updates += 1

        loss = sum(loss_info)

        # Collect total loss stats
        stats["loss"] = loss.detach()
        stats["generated_sample_prediction_loss"] = generated_sample_prediction_loss
        stats["real_sample_prediction_loss"] = real_sample_prediction_loss
        stats["gp"] = gp
        stats["sp"] = sp
        stats["pdl"] = pdl
        stats["mmi"] = mmi

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight, vocab_seen

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """
            Run inference on the given speech input to generate samples.

        This method extracts features from the input speech and uses the generator
        to create fake samples based on those features.

        Args:
            speech (torch.Tensor): A tensor containing the input speech signal.
            speech_lengths (torch.Tensor): A tensor containing the lengths of the
                input speech signals.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - generated_sample (torch.Tensor): The generated samples.
                - generated_sample_padding_mask (torch.Tensor): The padding mask
                  for the generated samples.

        Examples:
            >>> model = ESPnetUASRModel(...)  # Initialize the model
            >>> speech = torch.randn(2, 16000)  # Example speech input (batch size 2)
            >>> speech_lengths = torch.tensor([16000, 15000])  # Lengths of the inputs
            >>> generated_sample, padding_mask = model.inference(speech, speech_lengths)
            >>> print(generated_sample.shape)  # Output shape of generated samples
            >>> print(padding_mask.shape)  # Output shape of padding mask

        Note:
            Ensure that the model has been properly initialized and trained before
            calling this method for inference.
        """
        # 1. Feats encode (Extract feats + Apply segmenter)
        feats, padding_mask = self.encode(speech, speech_lengths)

        # 2. Generate fake samples
        (
            generated_sample,
            _,
            x_inter,
            generated_sample_padding_mask,
        ) = self.generator(feats, None, padding_mask)

        # generated_sample = generated_sample.softmax(-1)

        return generated_sample, generated_sample_padding_mask

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Collects features from the input speech tensor.

        This method processes the input speech tensor through a frontend if
        available, applying necessary transformations to extract features. If
        no frontend is defined, the original speech tensor is returned as
        features.

        Args:
            speech (torch.Tensor): Input speech tensor of shape
                (Batch, NSamples).
            speech_lengths (torch.Tensor): Lengths of the input speech tensor
                of shape (Batch,).
            text (Optional[torch.Tensor], optional): Input text tensor of
                shape (Batch, NText). Defaults to None.
            text_lengths (Optional[torch.Tensor], optional): Lengths of the
                input text tensor of shape (Batch,). Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'feats': Extracted features tensor of shape
                  (Batch, NFrames, Dim).
                - 'feats_lengths': Lengths of the extracted features tensor
                  of shape (Batch,).

        Examples:
            >>> model = ESPnetUASRModel(...)
            >>> speech = torch.randn(8, 16000)  # Batch of 8 samples
            >>> speech_lengths = torch.tensor([16000] * 8)  # All samples are
            >>> processed with length 16000
            >>> features = model.collect_feats(speech, speech_lengths)
            >>> print(features['feats'].shape)  # Expected shape: (8, NFrames, Dim)

        Note:
            The frontend processing may include operations such as STFT or
            other feature extraction methods.
        """
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            speech = F.layer_norm(speech, speech.shape)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None and not self.use_collected_training_feats:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            speech = F.layer_norm(speech, speech.shape)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract (usually with pre-extracted feat)
            # logging.info("use exisitng features")
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input speech tensor into features and create a padding mask.

        This method extracts features from the input speech tensor and applies
        a segmentation process if a segmenter is provided. It returns the
        extracted features along with a padding mask that indicates the valid
        elements in the feature tensor.

        Args:
            speech (torch.Tensor): Input speech tensor of shape
                (batch_size, num_samples).
            speech_lengths (torch.Tensor): Tensor containing the lengths of each
                speech sample in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feats (torch.Tensor): Extracted feature tensor of shape
                    (batch_size, num_frames, feature_dim).
                - padding_mask (torch.Tensor): Boolean tensor of shape
                    (batch_size, num_frames) indicating valid frames.

        Examples:
            >>> speech = torch.randn(8, 16000)  # 8 samples of 1 second each
            >>> speech_lengths = torch.tensor([16000] * 8)  # all samples are 1s
            >>> model = ESPnetUASRModel(...)
            >>> feats, padding_mask = model.encode(speech, speech_lengths)

        Note:
            The input speech tensor is expected to be of shape
            (batch_size, num_samples) where `num_samples` can vary. The
            lengths tensor should be a 1D tensor containing the actual lengths
            of each sample in the batch.

        Raises:
            AssertionError: If `speech_lengths` does not have the expected
            dimensions.
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        padding_mask = make_pad_mask(feats_lengths).to(feats.device)

        # 2. Apply feats
        if self.use_segmenter:
            feats, padding_mask = self.segmenter.pre_segment(feats, padding_mask)

        return feats, padding_mask

    def is_discriminative_step(self):
        """
            Determines whether the current update step is a discriminative step.

        This method checks the number of updates and returns `True` if the number
        of updates is odd, indicating that the current training iteration is a
        discriminative step, and `False` otherwise.

        Returns:
            bool: True if the current update is a discriminative step, otherwise False.

        Examples:
            >>> model = ESPnetUASRModel(...)  # Initialize the model with necessary args
            >>> model.number_updates = 1
            >>> model.is_discriminative_step()  # Returns True
            >>> model.number_updates = 2
            >>> model.is_discriminative_step()  # Returns False
        """
        return self.number_updates % 2 == 1

    def get_optim_index(self):
        """
            Get the optimization index based on the number of updates.

        This method calculates the optimization index used for alternating
        updates in the training process. It returns 0 or 1 depending on
        whether the number of updates is even or odd. This can be useful
        for implementing different training strategies based on the
        optimization step.

        Returns:
            int: The optimization index (0 or 1) determined by the
            current number of updates.

        Examples:
            >>> model = ESPnetUASRModel(...)
            >>> model.number_updates = 1
            >>> model.get_optim_index()
            1
            >>> model.number_updates = 2
            >>> model.get_optim_index()
            0
        """
        return self.number_updates % 2

    def _change_temperature(self):
        self.current_temperature = max(
            self.max_temperature * self.decay_temperature**self.number_updates,
            self.min_temperature,
        )
