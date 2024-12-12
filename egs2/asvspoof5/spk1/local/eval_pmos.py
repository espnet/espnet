#!/usr/bin/env python3

# borrowed from: https://github.com/tarepan/SpeechMOS/blob/main/speechmos/utmos22/fairseq_alt.py
"""W2V2 model optimized to UTMOS22 strong learner inference. Origin cloned from FairSeq under MIT license (Copyright Facebook, Inc. and its affiliates., https://github.com/facebookresearch/fairseq/blob/main/LICENSE)."""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio

class Wav2Vec2Model(nn.Module):
    """Wav2Vev2."""

    def __init__(self):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        feat_h1, feat_h2 = 512, 768
        feature_enc_layers = [(feat_h1, 10, 5)] + [(feat_h1, 3, 2)] * 4 + [(feat_h1, 2, 2)] * 2

        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers) # pyright: ignore [reportGeneralTypeIssues]
        self.layer_norm = nn.LayerNorm(feat_h1)
        self.post_extract_proj = nn.Linear(feat_h1, feat_h2)
        self.dropout_input = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(feat_h2)

        # Remnants
        self.mask_emb = nn.Parameter(torch.FloatTensor(feat_h2))

    def forward(self, source: Tensor):
        """FeatureEncoder + ContextTransformer"""

        # Feature encoding
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)

        # Context transformer
        x = self.encoder(features)

        return x


class ConvFeatureExtractionModel(nn.Module):
    """Feature Encoder."""

    def __init__(self, conv_layers: List[Tuple[int, int, int]]):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        def block(n_in: int, n_out: int, k: int, stride: int, is_group_norm: bool = False):
            if is_group_norm:
                return nn.Sequential(nn.Conv1d(n_in, n_out, k, stride=stride, bias=False), nn.Dropout(p=0.0), nn.GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                return nn.Sequential(nn.Conv1d(n_in, n_out, k, stride=stride, bias=False), nn.Dropout(p=0.0),                                      nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, params in enumerate(conv_layers):
            (dim, k, stride) = params
            self.conv_layers.append(block(in_d, dim, k, stride, is_group_norm = i==0))
            in_d = dim

    def forward(self, series: Tensor) -> Tensor:
        """ :: (B, T) -> (B, Feat, Frame)"""

        series = series.unsqueeze(1)
        for conv in self.conv_layers:
            series = conv(series)

        return series


class TransformerEncoder(nn.Module):
    """Transformer."""

    def build_encoder_layer(self, feat: int):
        """Layer builder."""
        return TransformerSentenceEncoderLayer(
            embedding_dim=feat,
            ffn_embedding_dim=3072,
            num_attention_heads=12,
            activation_fn="gelu",
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            layer_norm_first=False,
        )

    def __init__(self, feat: int):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        self.required_seq_len_multiple = 2

        self.pos_conv = nn.Sequential(*[
            nn.utils.weight_norm(nn.Conv1d(feat, feat, kernel_size=128, padding=128//2, groups=16), name="weight", dim=2),
            SamePad(128),
            nn.GELU()
        ])
        self.layer_norm = nn.LayerNorm(feat)
        self.layers = nn.ModuleList([self.build_encoder_layer(feat) for _ in range(12)])

    def forward(self, x: Tensor) -> Tensor:

        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)
        if pad_length > 0:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(None, self.required_seq_len_multiple, dim=-1, value=True)

        # :: (B, T, Feat) -> (T, B, Feat)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        # :: (T, B, Feat) -> (B, T, Feat)
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

        return x


class SamePad(nn.Module):
    """Tail inverse padding."""
    def __init__(self, kernel_size: int):
        super().__init__() # pyright: ignore [reportUnknownMemberType]
        assert kernel_size % 2 == 0, "`SamePad` now support only even kernel."

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, : -1]


def pad_to_multiple(x: Optional[Tensor], multiple: int, dim: int = -1, value: float = 0) -> Tuple[Optional[Tensor], int]:
    """Tail padding."""
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


class TransformerSentenceEncoderLayer(nn.Module):
    """Transformer Encoder Layer used in BERT/XLM style pre-trained models."""

    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        activation_fn: str,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        layer_norm_first: bool,
    ) -> None:
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        assert layer_norm_first == False, "`layer_norm_first` is fixed to `False`"
        assert activation_fn == "gelu", "`activation_fn` is fixed to `gelu`"

        feat = embedding_dim

        self.self_attn = MultiheadAttention(feat, num_attention_heads, attention_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feat, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, feat)
        self.self_attn_layer_norm = nn.LayerNorm(feat)
        self.final_layer_norm     = nn.LayerNorm(feat)

    def forward(self, x: Tensor, self_attn_padding_mask: Optional[Tensor]):
        # Res[Attn-Do]-LN
        residual = x
        x = self.self_attn(x, x, x, self_attn_padding_mask)
        x = self.dropout1(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # Res[SegFC-GELU-Do-SegFC-Do]-LN
        residual = x
        x = F.gelu(self.fc1(x)) # pyright: ignore [reportUnknownMemberType]
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class MultiheadAttention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        self.embed_dim, self.num_heads, self.p_dropout = embed_dim, num_heads, dropout
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            query            :: (T, B, Feat)
            key_padding_mask :: (B, src_len) - mask to exclude keys that are pads, where padding elements are indicated by 1s.
        """
        return F.multi_head_attention_forward(
            query = query,
            key   = key,
            value = value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.p_dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias  =self.out_proj.bias,
            training=False,
            key_padding_mask=key_padding_mask.bool() if key_padding_mask is not None else None,
            need_weights=False,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )[0]


# adapted from: https://github.com/tarepan/SpeechMOS/blob/main/speechmos/utmos22/strong/model.py
class UTMOS22Strong(nn.Module):
    """ Saeki_2022 paper's `UTMOS strong learner` inference model (w/o Phoneme encoder)."""

    def __init__(self):
        """Init."""

        super().__init__() # pyright: ignore [reportUnknownMemberType]

        feat_ssl, feat_domain_emb, feat_judge_emb, feat_rnn_h, feat_proj_h = 768, 128, 128, 512, 2048
        feat_cat = feat_ssl + feat_domain_emb + feat_judge_emb

        # SSL/DataDomainEmb/JudgeIdEmb/BLSTM/Projection
        self.wav2vec2 = Wav2Vec2Model()
        self.domain_emb = nn.Parameter(data=torch.empty(1, feat_domain_emb), requires_grad=False)
        self.judge_emb  = nn.Parameter(data=torch.empty(1, feat_judge_emb),  requires_grad=False)
        self.blstm = nn.LSTM(input_size=feat_cat, hidden_size=feat_rnn_h, batch_first=True, bidirectional=True)
        # self.projection = nn.Sequential(nn.Linear(feat_rnn_h*2, feat_proj_h), nn.ReLU(), nn.Linear(feat_proj_h, 1))
        self.proj_1 = nn.Linear(feat_rnn_h*2, feat_proj_h)
        self.proj_relu = nn.ReLU()
        self.proj_2 = nn.Linear(feat_proj_h, 1)

    def forward(self, wave: Tensor, sr: int, return_feat: bool = False) -> Tensor: # pylint: disable=invalid-name
        """wave-to-score :: (B, T) -> (B,) """

        # Resampling :: (B, T) -> (B, T)
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)

        # Feature extraction :: (B, T) -> (B, Frame, Feat)
        unit_series = self.wav2vec2(wave)
        bsz, frm, _ = unit_series.size()

        # DataDomain/JudgeId Embedding's Batch/Time expansion :: (B=1, Feat) -> (B=bsz, Frame=frm, Feat)
        domain_series = self.domain_emb.unsqueeze(1).expand(bsz, frm, -1)
        judge_series  =  self.judge_emb.unsqueeze(1).expand(bsz, frm, -1)

        # Feature concatenation :: (B, Frame, Feat=f1) + (B, Frame, Feat=f2) + (B, Frame, Feat=f3) -> (B, Frame, Feat=f1+f2+f3)
        cat_series = torch.cat([unit_series, domain_series, judge_series], dim=2)

        # Frame-scale score estimation :: (B, Frame, Feat) -> (B, Frame, Feat) -> (B, Frame, Feat=1) - BLSTM/Projection
        feat_series = self.blstm(cat_series)[0]

        feat_series = self.proj_1(feat_series)
        if return_feat:
            # return a frame-level feature
            return feat_series
        feat_series = self.proj_relu(feat_series)
        feat_series = self.proj_2(feat_series)
        score_series = self.projection(feat_series)

        # Utterance-scale score :: (B, Frame, Feat=1) -> (B, Feat=1) -> (B,) - Time averaging
        utter_score = score_series.mean(dim=1).squeeze(1) * 2 + 3

        return utter_score

# Copyright 2023 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate pseudo MOS calculated by automatic MOS prediction model."""

import argparse
import fnmatch
import logging
import os
from typing import List

import librosa
import numpy as np
import torch


def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> List[str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        List[str]: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def calculate(
    file_list: List[str],
    predictor: torch.nn.Module,
    device: torch.device,
    batchsize: int,
    outdir: str = "exp",
    wavscp_dict: dict = None,
):
    """Calculate frame features using wav.scp mapping"""
    fs = librosa.get_samplerate(list(wavscp_dict.values())[0])
    files = list(wavscp_dict.items())  # list of (uttid, path) pairs

    feat_dir = os.path.join(outdir, "frame_features")
    os.makedirs(feat_dir, exist_ok=True)

    for si in range(0, len(files), batchsize):
        batch_files = files[si:si + batchsize]
        uttids = [x[0] for x in batch_files]  # Get uttids for this batch
        paths = [x[1] for x in batch_files]   # Get paths for this batch
        
        # Load audio
        gen_xs = []
        for path in paths:
            gen_x, gen_fs = librosa.load(path, sr=None, mono=True)
            assert fs == gen_fs
            gen_xs.append(gen_x)
            
        # Padding
        max_len = max([len(gen_x) for gen_x in gen_xs])
        gen_xs = [np.pad(x, (0, max_len - len(x)), "constant", constant_values=0) for x in gen_xs]
        gen_xs = torch.from_numpy(np.stack(gen_xs)).to(device)

        # Get frame-level features
        with torch.no_grad():
            frame_feats = predictor(gen_xs, fs, return_feat=True)  # [B, Frames, 2048]
            frame_feats = frame_feats.cpu().numpy()
            
            # Save features using uttids directly
            for uttid, feat in zip(uttids, frame_feats):
                feat_path = os.path.join(feat_dir, f"{uttid}.npy")
                np.save(feat_path, feat)
                logging.info(f"Saved features for {uttid} with shape {feat.shape}")
    return    

def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate pseudo MOS.")
    parser.add_argument(
        "gen_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument(
        "--mos_toolkit",
        type=str,
        default="utmos",
        choices=["utmos"],
        help="Toolkit to calculate pseudo MOS.",
    )
    parser.add_argument(
        "--batchsize",
        default=4,
        type=int,
        help="Number of batches.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run pseudo MOS calculation in parallel."""
    args = get_parser().parse_args()

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # Find files for generated speech waveforms.
    if os.path.isdir(args.gen_wavdir_or_wavscp):
        gen_files = sorted(find_files(args.gen_wavdir_or_wavscp))
    else:
        with open(args.gen_wavdir_or_wavscp) as f:
            gen_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gen_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")
        
    # Create wav.scp dictionary
    wavscp_dict = {}
    with open(args.gen_wavdir_or_wavscp) as f:
        for line in f:
            uttid, path = line.strip().split(None, 1)
            wavscp_dict[uttid] = path

    logging.info("The number of utterances = %d" % len(wavscp_dict))

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    logging.info("The number of utterances = %d" % len(gen_files))

    if torch.cuda.is_available() and ("cuda" in args.device):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    predictor = UTMOS22Strong().to(device)

    if args.mos_toolkit == "utmos":
        # Load predictor for UTMOS22.
        loaded_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong").to(
            device
        )
        loaded_state_dict = loaded_predictor.state_dict()
        # Create new state dict with remapped keys
        new_state_dict = {}
        for key, value in loaded_state_dict.items():
            if key == 'projection.0.weight':
                new_state_dict['proj_1.weight'] = value
            elif key == 'projection.0.bias':
                new_state_dict['proj_1.bias'] = value
            elif key == 'projection.2.weight':
                new_state_dict['proj_2.weight'] = value
            elif key == 'projection.2.bias':
                new_state_dict['proj_2.bias'] = value
            else:
                new_state_dict[key] = value

        predictor.load_state_dict(new_state_dict)
        predictor.eval()
    else:
        raise NotImplementedError(f"Not supported {args.mos_toolkit}.")

    calculate(gen_files, predictor, device, args.batchsize, args.outdir, wavscp_dict)

    logging.info("Successfully finished pseudo MOS extraction.")


if __name__ == "__main__":
    main()
