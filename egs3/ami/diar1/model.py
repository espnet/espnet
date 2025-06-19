from egs3.ami.diar1.local.hungarian_pitloss import PITLossWrapper
from egs3.ami.diar1.local.utils import len2mask
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from typeguard import typechecked
from espnet2.torch_utils.device_funcs import force_gatherable
from torch.cuda.amp import autocast
from egs3.ami.diar1.local.utils import padtrunc
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from speechbrain.lobes.models.conv_tasnet import ChannelwiseLayerNorm


class ConvFrontEnd(torch.nn.Module):
    def __init__(self,
            n_mels=23,
            in_ksz=15,
            in_stride=10,
            emb_size=256,
            dropout=0.0,
            pos_enc_ksz=31,
            eps=1e-5):
        super().__init__()

        latent_dim = int(n_mels*in_ksz)
        self.depthwise = torch.nn.Conv1d(n_mels, latent_dim, kernel_size=in_ksz,
                                     stride=in_stride, padding=in_ksz // 2, groups=n_mels)
        # expand here
        self.in_norm = ChannelwiseLayerNorm(latent_dim)
        self.pointwise = torch.nn.Linear(latent_dim, emb_size) # bottleneck

        self.in_drop = torch.nn.Dropout(dropout)
        self.pos_enc = torch.nn.Conv1d(emb_size, emb_size, kernel_size=pos_enc_ksz,
                                         stride=1, padding=pos_enc_ksz // 2, groups=emb_size)


    def forward(self, x, lengths):
        x = self.depthwise(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.pointwise(self.in_norm(x))
        #x = self.in_drop(x)
        x = self.pos_enc(x.transpose(-1, -2)).transpose(-1, -2)

        ratio =  x.shape[1] / lengths
        adjusted_lengths = (lengths * ratio).long()

        return x, adjusted_lengths

class UpSamplingDecoder(torch.nn.Module):
    def __init__(self, n_local_spk=5,
                 emb_size=256,
                 ksz=[3, 5],
                 strides=[2, 5],
                 eps=1e-5):
        super().__init__()

        self.upsampling1 = torch.nn.ConvTranspose1d(emb_size, emb_size, ksz[0], stride=strides[0])
        self.upnorm1 = torch.nn.LayerNorm(emb_size, eps=eps)
        self.upsampling2 = torch.nn.ConvTranspose1d(emb_size, emb_size, ksz[1], stride=strides[1])
        self.upnorm2 = torch.nn.LayerNorm(emb_size, eps=eps)
        self.out_proj = torch.nn.Linear(emb_size, n_local_spk)

    def forward(self, x):

        x = self.upsampling1(x.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.gelu(self.upnorm1(x))
        x = self.upsampling2(x.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.gelu(self.upnorm2(x))
        x = self.out_proj(x)
        return x


class MaskedBinaryXentropyWithLogits(torch.nn.Module):
    def __init__(self, balance_spk=False):
        super().__init__()
        self.balance_spk = balance_spk

    def forward(self, logits, targets, lengths):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        if lengths is None:
            pass
        else:
            mask = ~len2mask(lengths, targets.shape[-1], torch.bool)
            loss = loss.masked_fill(mask[:, None], 0.0)
        # not sure if we should boost the speakers for which the activity is low
        # probably yes for ASR, but DER and JER won t measure it.
        if self.balance_spk:
            # get weights for each speaker
            pos_ratio = targets.sum(dim=-1, keepdim=True) / targets.shape[-1]  # [batch, speakers, 1]
            pos_ratio = torch.clamp(pos_ratio, min=0.01, max=0.99)  # Avoid extreme values

            # Weight inversely proportional to class frequency
            weights = torch.where(targets == 1,
                                  1.0 / (pos_ratio + 1e-8),
                                  1.0 / (1.0 - pos_ratio + 1e-8))

            loss = loss.sum(-1) * weights[..., 0]
        else:
            loss = loss.sum(-1) # we should probably divide here

        return loss.mean(1)


class VanillaEENDModelWrapper(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        frontend,
        encoder,
        decoder,
        lossfunc,
        log_der_train: Optional[bool] = True
    ):
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.log_der_train = log_der_train
        # should we also use DICE loss ?
        self.pit_loss = PITLossWrapper(lossfunc, pit_from="perm_avg")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, seq_len, feat_dim)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            spk_labels: (Batch, seq_len)
            kwargs: "utt_id" is among the input.
        """

        batch_size = speech.size(0)
        orig_len = speech.shape[1]

        if not self.training:
            with torch.inference_mode():
                speech, speech_lengths = self.encode(speech, speech_lengths)
                speech = self.decoder(speech)
        else:
            speech, speech_lengths = self.encode(speech, speech_lengths)
            speech = self.decoder(speech)

        speech = speech[:, :orig_len]

        lendiff = abs(speech.shape[1] - spk_labels.shape[1])
        if  lendiff != 0 and lendiff <= 5:
            speech = padtrunc(speech, spk_labels.shape[1])

        loss, speech = self.compute_loss(speech.transpose(1, 2), spk_labels.transpose(1, 2).float(), spk_labels_lengths)
        speech = speech.transpose(1, 2) # since it is fed transposed

        loss = loss.mean()

        with torch.cuda.amp.autocast(enabled=False):
            speech = speech.float()
            spk_labels = spk_labels.float()
            if self.log_der_train or not self.training:
                (correct,
                 num_frames,
                 speech_scored,
                 speech_miss,
                 speech_falarm,
                 speaker_scored,
                 speaker_miss,
                 speaker_falarm,
                 speaker_error,
                 ) = ESPnetDiarizationModel.calc_diarization_error(speech,
                                                 spk_labels,
                                                 spk_labels_lengths)

                if speech_scored > 0 and num_frames > 0:
                    sad_mr, sad_fr, mi, fa, cf, acc, der = (
                        speech_miss / speech_scored,
                        speech_falarm / speech_scored,
                        speaker_miss / speaker_scored,
                        speaker_falarm / speaker_scored,
                        speaker_error / speaker_scored,
                        correct / num_frames,
                        (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
                    )
                else:
                    sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

                stats = dict(
                    loss=loss.detach(),
                    sad_mr=sad_mr,
                    sad_fr=sad_fr,
                    mi=mi,
                    fa=fa,
                    cf=cf,
                    acc=acc,
                    der=der,
                )
            else:
                stats = dict(
                    loss=loss.detach(),
                )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder
        Args:
            feats: (Batch, Length, ...)
            feats_lengths: (Batch,)
        """

        with autocast(False):
            # note that for efficiency the feature extraction is moved in the dataloader
            feats, feats_lengths = self.frontend(feats, feats_lengths)

            if not self.training:
                assert feats.shape[0] == 1
                feats_lengths = torch.tensor([feats.shape[1]]).to(feats.device)

            feats, feats_lengths, _ = self.encoder(feats, feats_lengths)

        return feats, feats_lengths

    def compute_loss(self, pred, label, length):

        loss, batch_indx = self.pit_loss(pred, label, lengths=length, return_est=True)
        pred = PITLossWrapper.reorder_source(pred, batch_indx)
        return loss, pred
