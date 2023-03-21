import itertools
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel as SingleESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class PITLossWrapper(AbsLossWrapper):
    def __init__(self, criterion_fn: Callable, num_ref: int):
        super().__init__()
        self.criterion_fn = criterion_fn
        self.num_ref = num_ref

    def forward(
        self,
        inf: torch.Tensor,
        inf_lens: torch.Tensor,
        ref: torch.Tensor,
        ref_lens: torch.Tensor,
        others: Dict = None,
    ):
        """PITLoss Wrapper function. Similar to espnet2/enh/loss/wrapper/pit_solver.py

        Args:
            inf: Iterable[torch.Tensor], (batch, num_inf, ...)
            inf_lens: Iterable[torch.Tensor], (batch, num_inf, ...)
            ref: Iterable[torch.Tensor], (batch, num_ref, ...)
            ref_lens: Iterable[torch.Tensor], (batch, num_ref, ...)
            permute_inf: If true, permute the inference and inference_lens according to
                the optimal permutation.
        """
        assert (
            self.num_ref
            == inf.shape[1]
            == inf_lens.shape[1]
            == ref.shape[1]
            == ref_lens.shape[1]
        ), (self.num_ref, inf.shape, inf_lens.shape, ref.shape, ref_lens.shape)

        all_permutations = torch.as_tensor(
            list(itertools.permutations(range(self.num_ref), r=self.num_ref))
        )

        stats = defaultdict(list)

        def pre_hook(func, *args, **kwargs):
            ret = func(*args, **kwargs)
            for k, v in getattr(self.criterion_fn, "stats", {}).items():
                stats[k].append(v)
            return ret

        def pair_loss(permutation):
            return sum(
                [
                    pre_hook(
                        self.criterion_fn,
                        inf[:, j],
                        inf_lens[:, j],
                        ref[:, i],
                        ref_lens[:, i],
                    )
                    for i, j in enumerate(permutation)
                ]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in all_permutations], dim=1
        )  # (batch_size, num_perm)

        min_losses, min_ids = torch.min(losses, dim=1)
        opt_perm = all_permutations[min_ids]  # (batch_size, num_ref)

        # Permute the inf and inf_lens according to the optimal perm
        return min_losses.mean(), opt_perm

    @classmethod
    def permutate(self, perm, *args):
        ret = []
        batch_size = None
        num_ref = None

        for arg in args:  # (batch, num_inf, ...)
            if batch_size is None:
                batch_size, num_ref = arg.shape[:2]
            else:
                assert torch.Size([batch_size, num_ref]) == arg.shape[:2]

            ret.append(
                torch.stack(
                    [arg[torch.arange(batch_size), perm[:, i]] for i in range(num_ref)],
                    dim=1,
                )
            )
        return ret


class ESPnetASRModel(SingleESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

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
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        # num_inf: the number of inferences (= number of outputs of the model)
        # num_ref: the number of references (= number of groundtruth seqs)
        num_inf: int = 1,
        num_ref: int = 1,
    ):
        assert check_argument_types()
        assert 0.0 < ctc_weight <= 1.0, ctc_weight
        assert interctc_weight == 0.0, "interctc is not supported for multispeaker ASR"

        super(ESPnetASRModel, self).__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        assert num_inf == num_ref, "Current PIT loss wrapper requires num_inf=num_ref"
        self.num_inf = num_inf
        self.num_ref = num_ref

        self.pit_ctc = PITLossWrapper(criterion_fn=self.ctc, num_ref=num_ref)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
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
        text_ref = [text] + [
            kwargs["text_spk{}".format(spk + 1)] for spk in range(1, self.num_ref)
        ]
        text_ref_lengths = [text_lengths] + [
            kwargs.get("text_spk{}_lengths".format(spk + 1), None)
            for spk in range(1, self.num_ref)
        ]

        assert all(ref_lengths.dim() == 1 for ref_lengths in text_ref_lengths), (
            ref_lengths.shape for ref_lengths in text_ref_lengths
        )

        text_lengths = torch.stack(text_ref_lengths, dim=1)  # (batch, num_ref)
        text_length_max = text_lengths.max()
        # pad text sequences of different speakers to the same length
        text = torch.stack(
            [
                torch.nn.functional.pad(
                    ref, (0, text_length_max - ref.shape[1]), value=self.ignore_id
                )
                for ref in text_ref
            ],
            dim=1,
        )  # (batch, num_ref, seq_len)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            # CTC is computed twice
            # This 1st ctc calculation is only used to decide permutation
            _, perm = self.pit_ctc(encoder_out, encoder_out_lens, text, text_lengths)
            encoder_out, encoder_out_lens = PITLossWrapper.permutate(
                perm, encoder_out, encoder_out_lens
            )
            if text.dim() == 3:  # combine all speakers hidden vectors and labels.
                encoder_out = encoder_out.reshape(-1, *encoder_out.shape[2:])
                encoder_out_lens = encoder_out_lens.reshape(-1)
                text = text.reshape(-1, text.shape[-1])
                text_lengths = text_lengths.reshape(-1)

            # This 2nd ctc calculation is to compute the loss
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            loss_ctc = loss_ctc.sum()

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

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
