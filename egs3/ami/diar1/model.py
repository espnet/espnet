from egs3.ami.diar1.local.hungarian_pitloss import PITLossWrapper
from egs3.ami.diar1.local.utils import len2mask
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from typeguard import typechecked
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.torch_utils.device_funcs import force_gatherable
from torch.cuda.amp import autocast
#from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder

class ConvFrontEnd(torch.nn.Module):
    def __init__(self,
            n_mels=23,
            in_ksz=15,
            in_stride=10,
            emb_size=256,
            dropout=0.0,
            eps=1e-5):
        super().__init__()

        self.depthwise = torch.nn.Conv1d(n_mels, n_mels, kernel_size=in_ksz,
                                     stride=in_stride, padding=in_ksz // 2, groups=n_mels)
        self.pointwise = torch.nn.Linear(n_mels, emb_size)
        self.in_norm = torch.nn.LayerNorm(emb_size, eps=eps)
        self.in_drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.pointwise(self.depthwise(x.transpose(-1, -2)).transpose(-1, -2))
        x = self.in_drop(self.in_norm(x))
        return x

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
        mask = ~len2mask(lengths, targets.shape[1])
        loss = loss.masked_fill(mask, 0.0)
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

            loss = loss.sum(-1) * weights
        else:
            loss = loss.sum(-1)

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
        import pdb
        pdb.set_trace()
        batch_size = speech.size(0)
        speech, speech_lengths = self.encode(speech, speech_lengths)
        speech = self.decoder(speech)
        loss, speech = self.compute_loss(speech, spk_labels, spk_labels_lengths)

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
            ) = self.calc_diarization_error(torch.sigmoid(speech),
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
            feats, feats_lengths, _ = self.encoder(feats, feats_lengths)

        return feats, feats_lengths

    def compute_loss(self, pred, label, length):

        loss, batch_indx = self.pit_loss(pred, label, length=length, return_est=True)
        pred = PITLossWrapper.reorder_source(pred, batch_indx)
        return loss, pred


    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0.5).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )