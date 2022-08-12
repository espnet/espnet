import logging
import random
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

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
    """Joint model Enhancement and Speech to Text."""

    def __init__(
        self,
        enh_model: ESPnetEnhancementModel,
        s2t_model: Union[ESPnetASRModel, ESPnetSTModel, ESPnetDiarizationModel],
        calc_enh_loss: bool = True,
        bypass_enh_prob: float = 0,  # 0 means do not bypass enhancement for all data
    ):
        assert check_argument_types()

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

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, ) default None for chunk interator,
                                      because the chunk-iterator does not
                                      have the speech_lengths returned.
                                      see in
                                      espnet2/iterators/chunk_iter_factory.py
            text: (Batch, Length) default None just to keep the argument order
            text_lengths: (Batch,) default None for the same reason as speech_lengths
        """
        if text_lengths is not None:
            assert text_lengths.dim() == 1, text_lengths.shape
        if speech_lengths is not None and text_lengths is not None:
            # Check that batch_size is unified
            assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
            ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        else:
            assert speech.shape[0] == text.shape[0], (speech.shape, text.shape)

        # additional checks with valid src_text
        if "src_text" in kwargs:
            src_text = kwargs["src_text"]
            src_text_lengths = kwargs["src_text_lengths"]

            if src_text is not None:
                assert src_text_lengths.dim() == 1, src_text_lengths.shape
                assert (
                    text.shape[0] == src_text.shape[0] == src_text_lengths.shape[0]
                ), (
                    text.shape,
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
            if utt_id[0].endswith("SIMU"):
                # For simulated single-/multi-speaker data
                # feed it to Enhancement and calculate loss_enh
                bypass_enh_flag = False
                skip_enhloss_flag = False
            elif utt_id[0].endswith("REAL"):
                # For single-speaker real data
                # feed it to Enhancement but without calculating loss_enh
                bypass_enh_flag = False
                skip_enhloss_flag = True
            else:
                # For clean data
                # feed it to Enhancement, without calculating loss_enh
                bypass_enh_flag = True
                skip_enhloss_flag = True

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
        if not bypass_enh_flag:
            (
                speech_pre,
                feature_mix,
                feature_pre,
                others,
            ) = self.enh_model.forward_enhance(
                speech, speech_lengths, {"num_spk": num_spk}
            )
            # loss computation
            if not skip_enhloss_flag:
                loss_enh, _, _ = self.enh_model.forward_loss(
                    speech_pre,
                    speech_lengths,
                    feature_mix,
                    feature_pre,
                    others,
                    speech_ref,
                )
                loss_enh = loss_enh[0]
        else:
            speech_pre = [speech]

        # for data-parallel
        if text_lengths is not None:
            text = text[:, : text_lengths.max()]
        if src_text is not None:
            src_text = src_text[:, : src_text_lengths.max()]

        # 2. ASR or ST
        if isinstance(self.s2t_model, ESPnetASRModel):  # ASR
            loss_asr, stats, weight = self.s2t_model(
                speech_pre[0], speech_lengths, text, text_lengths
            )
        elif isinstance(self.s2t_model, ESPnetSTModel):  # ST
            loss_asr, stats, weight = self.s2t_model(
                speech_pre[0],
                speech_lengths,
                text,
                text_lengths,
                src_text,
                src_text_lengths,
            )
        elif isinstance(self.s2t_model, ESPnetDiarizationModel):  # DIAR
            loss_asr, stats, weight = self.s2t_model(
                speech=speech.clone(),
                speech_lengths=speech_lengths,
                spk_labels=text,
                spk_labels_lengths=text_lengths,
                bottleneck_feats=others.get("bottleneck_feats"),
                bottleneck_feats_lengths=others.get("bottleneck_feats_lengths"),
            )
        else:
            raise NotImplementedError(f"{type(self.s2t_model)} is not supported yet.")

        if loss_enh is not None:
            loss = loss_enh + loss_asr
        else:
            loss = loss_asr

        stats["loss"] = loss.detach() if loss is not None else None
        stats["loss_enh"] = loss_enh.detach() if loss_enh is not None else None

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
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
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        (
            speech_pre,
            feature_mix,
            feature_pre,
            others,
        ) = self.enh_model.forward_enhance(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.s2t_model.encode(
            speech_pre[0], speech_lengths
        )

        return encoder_out, encoder_out_lens

    def encode_diar(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, num_spk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by diar_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            num_spk: int
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
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        return self.s2t_model.nll(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens,
        )

    batchify_nll = ESPnetASRModel.batchify_nll

    def inherite_attributes(
        self,
        inherite_enh_attrs: List[str] = [],
        inherite_s2t_attrs: List[str] = [],
    ):
        assert check_argument_types()

        if len(inherite_enh_attrs) > 0:
            for attr in inherite_enh_attrs:
                setattr(self, attr, getattr(self.enh_model, attr, None))
        if len(inherite_s2t_attrs) > 0:
            for attr in inherite_s2t_attrs:
                setattr(self, attr, getattr(self.s2t_model, attr, None))
