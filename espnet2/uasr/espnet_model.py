import logging
import argparse

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import kenlm
import editdistance

import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.text.token_id_converter import TokenIDConverter

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.uasr.generator.abs_generator import AbsGenerator
from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter

from espnet2.uasr.loss.abs_loss import AbsUASRLoss

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.utils.types import str2bool

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetUASRModel(AbsESPnetModel):
    """
    Unsupervised ASR model
    The source code is from FAIRSEQ: 
        https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/unsupervised
    """

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
        skip_softmax: str2bool = False,
        use_gumbel: str2bool = False,
        use_hard_gumbel: str2bool = True,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0,
        decay_temperature: float = 0.99995,
        use_collected_training_feats: str2bool = False,
    ):
        assert check_argument_types()

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
        self.ll_increasing_type = "constant"
        if ll_increasing_type is not None:
            self.ll_increasing_type = ll_increasing_type

        # for validation
        self.vocab_size = vocab_size
        self.token_list = token_list
        self.token_id_converter = TokenIDConverter(token_list=token_list)
        self.sil_id = self.token_id_converter.tokens2ids([sil_token])[0]

        self.kenlm = None
        if kenlm_path:
            self.kenlm = kenlm.Model(kenlm_path)

    @property
    def number_updates(self):
        return self._number_updates

    @number_updates.setter
    def number_updates(self, iiter: int):
        assert check_argument_types() and iiter >= 0
        self._number_updates = iiter

    @property
    def number_epochs(self):
        return self._number_epochs

    @number_epochs.setter
    def number_epochs(self, iepoch: int):
        assert check_argument_types() and iepoch >= 1
        self._number_epochs = iepoch

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
        """Frontend + Segmenter + Generator + Discriminator + Calc Loss

        Args:
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
                hyp_ids_nosil = hyp_ids[hyp_ids != self.sil_id]
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
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        speech = F.layer_norm(speech, speech.shape)
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
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
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract (usually with pre-extracted feat)
            # logging.info("use exisitng features")
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        padding_mask = make_pad_mask(feats_lengths).to(feats.device)

        # 2. Apply feats
        if self.use_segmenter:
            feats, padding_mask = self.segmenter.pre_segment(feats, padding_mask)

        return feats, padding_mask

    def is_discriminative_step(self):
        return self.number_updates % 2 == 1

    def get_optim_index(self):
        return self.number_updates % 2

    def _change_temperature(self):
        self.current_temperature = max(
            self.max_temperature * self.decay_temperature**self.number_updates,
            self.min_temperature,
        )
