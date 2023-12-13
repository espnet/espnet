import contextlib
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from filelock import FileLock
from torch import nn
from typeguard import check_argument_types

from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.scorer_interface import BatchScorerInterface


class FairseqUnitBart(AbsSynthesizer, BatchScorerInterface):
    """FairSeq BART decoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        bart_url: url to mBART pretrained model
        bart_dir_path: directory to download the mBART pretrained model.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        bart_url: str,
        dict_url: str,
        bart_dir_path: str = "./",
        padding_idx: int = -1,
    ):
        assert check_argument_types()
        super().__init__()

        self._output_size = odim
        self.padding_idx = padding_idx

        try:
            import fairseq
            from fairseq.checkpoint_utils import load_model_ensemble_and_task
            from fairseq.models.bart import BARTModel
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        self.bart_model_path = download_bart(bart_url, bart_dir_path, "unit_mbart.pth")
        self.dict_path = download_bart(dict_url, bart_dir_path, "dict.txt")

        models, _, _ = load_model_ensemble_and_task(
            [self.bart_model_path],
            arg_overrides={"data": bart_dir_path},
        )
        model = models[0]

        if not isinstance(model, BARTModel):
            raise RuntimeError(
                "Error: pretrained models should be within: "
                "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
            )

        self.decoder = model.decoder
        self.pretrained_params = copy.deepcopy(model.decoder.state_dict())

        # FairSeq TransformerDecoder uses padding idx to construct the self attention mask
        self.decoder.padding_idx = self.padding_idx

        self.output_layer = nn.Linear(model.cfg.decoder_output_dim, odim)

    def output_size(self) -> int:
        return self._output_size

    def reload_pretrained_parameters(self):
        """
        Make sure FairSeq's pretrained weights are loaded correctly before training
        """
        self.decoder.load_state_dict(self.pretrained_params)
        logging.info("Pretrained unit mBART model parameters reloaded!")

    def forward(
        self,
        enc_out: torch.Tensor,
        enc_out_lens: Optional[torch.Tensor],
        ys: torch.Tensor,
        ys_lens: Optional[torch.Tensor],
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward mBART decoder.

        Args:
            enc_out (LongTensor): Batch of padded character ids (B, T, idim).
            enc_out_lens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, T_feats, odim).
            ys_lens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).

        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, vocab)
            olens: (batch, )

        See fairseq.models.transformer.transformer_decoder.TransformerDecoderBase
        """

        encoder_padding_mask = []
        if enc_out_lens is not None:
            encoder_padding_mask = [
                make_pad_mask(enc_out_lens, maxlen=enc_out.size(1)).to(enc_out.device)
            ]

        encoder_out = {
            "encoder_out": [enc_out.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        # FairSeq TransformerDecoder uses padding idx to construct the self attention mask
        # We don't have to pass it explicitly
        x, extra = self.decoder(
            prev_output_tokens=ys,
            encoder_out=encoder_out,
            features_only=True,  # we don't use pre-trained output layer
        )
        x = self.output_layer(x)

        intermediate_outs = extra["inner_states"]
        if return_hs:
            # TODO(Jiyang): this is the hidden state before LayerNorm,
            #   technically not what we want?
            return (x, intermediate_outs[-1]), ys_lens
        elif return_all_hs:
            return (x, intermediate_outs), ys_lens
        return x, ys_lens

    def score(self, ys, state, x):
        logp, _ = self.forward(
            ys.unsqueeze(0),
            None,
            x.unsqueeze(0),
            None,
        )
        return logp.squeeze(0), None

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        logp, _ = self.forward(
            ys,
            None,
            xs,
            None,
        )
        return logp, []

    def inference(
        self, input_states: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Not used for this type of synthesizer
        """
        pass


def download_bart(model_url: str, dir_path: str, out_filename: str):
    """
    Download model from `model_url` to `dir_path/out_filename`
    """

    os.makedirs(dir_path, exist_ok=True)

    model_path = os.path.join(dir_path, out_filename)

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            logging.info(f"BART model downloaded {model_path}")
        else:
            logging.info(f"BART model {model_path} already exists.")

    return model_path
