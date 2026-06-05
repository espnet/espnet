"""Convert a full NeMo SortformerEncLabelModel state-dict into this port.

Handles the **streaming** model ``nvidia/diar_streaming_sortformer_4spk-v2``
(distributed as ``.nemo`` only). Export the weights once in the NeMo env::

    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2",
                                                map_location="cpu")
    import torch; torch.save(m.state_dict(), "sortformer_v2_full.pt")

Then :func:`convert_nemo` builds a matching model and loads the weights.

Key remaps (NeMo -> this port):
  encoder.pre_encode.conv          -> encoder.subsampling.layers
  encoder.pre_encode.out           -> encoder.subsampling.linear
  encoder...self_attn.linear_q/k/v/out -> q_proj/k_proj/v_proj/o_proj
  encoder...self_attn.linear_pos   -> relative_k_proj ; pos_bias_u/v -> bias_u/v
  encoder...conv.batch_norm        -> conv.norm
  transformer_encoder...first_sub_layer.query_net/key_net/value_net/out_projection
                                   -> self_attn.q_proj/k_proj/v_proj/out_proj
  transformer_encoder...layer_norm_1 -> self_attn_layer_norm
  transformer_encoder...second_sub_layer.dense_in/dense_out -> fc1/fc2
  transformer_encoder...layer_norm_2 -> final_layer_norm
  sortformer_modules.*             -> sortformer_modules.* (unchanged)

The transformer key projection has no bias in this port (a softmax no-op), so the
NeMo ``key_net.bias`` is dropped.
"""

import torch

from espnet2.diar.espnet_sortformer_model import ESPnetSortformerModel
from espnet2.diar.sortformer.fastconformer_encoder import FastConformerEncoder
from espnet2.diar.sortformer.preprocessor import MelSpectrogramPreprocessor
from espnet2.diar.sortformer.sortformer_modules import SortformerModules
from espnet2.diar.sortformer.transformer_encoder import TransformerEncoder

# Streaming v2 hyper-parameters (encoder has 17 layers; NeMo SortformerModules
# defaults except the overrides in the v2 config).
V2 = dict(
    fc_n_layers=17,
    n_mels=128,  # v2 uses 128 mel bins (pre_encode.out is 512x4096 = 256*16)
    spkcache_sil_frames_per_spk=3,
    pred_score_threshold=0.25,
    sil_threshold=0.2,
    scores_boost_latest=0.05,
    scores_add_rnd=0.0,
    strong_boost_rate=0.75,
    weak_boost_rate=1.5,
    min_pos_scores_rate=0.5,
    max_index=99999,
)


def build_streaming_v2_model(num_spk: int = 4):
    # v2 preprocessor uses normalize="NA" (no per-feature normalization).
    pre = MelSpectrogramPreprocessor(
        sample_rate=16000, features=V2["n_mels"], normalize="NA"
    )
    enc = FastConformerEncoder(
        feat_in=V2["n_mels"],
        d_model=512,
        n_layers=V2["fc_n_layers"],
        n_heads=8,
        ff_expansion_factor=4,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        conv_kernel_size=9,
        dropout=0.1,
        dropout_att=0.1,
    )
    mods = SortformerModules(
        num_spks=num_spk,
        dropout_rate=0.5,
        fc_d_model=512,
        tf_d_model=192,
        spkcache_len=188,
        fifo_len=0,
        chunk_len=188,
        spkcache_update_period=188,
        chunk_left_context=1,
        chunk_right_context=1,
        spkcache_sil_frames_per_spk=V2["spkcache_sil_frames_per_spk"],
        pred_score_threshold=V2["pred_score_threshold"],
        sil_threshold=V2["sil_threshold"],
        scores_boost_latest=V2["scores_boost_latest"],
        scores_add_rnd=V2["scores_add_rnd"],
        strong_boost_rate=V2["strong_boost_rate"],
        weak_boost_rate=V2["weak_boost_rate"],
        min_pos_scores_rate=V2["min_pos_scores_rate"],
        max_index=V2["max_index"],
    )
    tf = TransformerEncoder(
        num_layers=18,
        hidden_size=192,
        inner_size=768,
        num_attention_heads=8,
        attn_score_dropout=0.5,
        attn_layer_dropout=0.5,
        ffn_dropout=0.5,
    )
    return ESPnetSortformerModel(pre, enc, mods, tf, num_spk=num_spk)


def remap_key(k: str):
    if k.startswith("preprocessor."):
        return None  # recomputed buffers
    if k.startswith("encoder."):
        nk = k
        nk = nk.replace("encoder.pre_encode.conv.", "encoder.subsampling.layers.")
        nk = nk.replace("encoder.pre_encode.out.", "encoder.subsampling.linear.")
        nk = nk.replace(".conv.batch_norm.", ".conv.norm.")
        nk = nk.replace(".self_attn.linear_q.", ".self_attn.q_proj.")
        nk = nk.replace(".self_attn.linear_k.", ".self_attn.k_proj.")
        nk = nk.replace(".self_attn.linear_v.", ".self_attn.v_proj.")
        nk = nk.replace(".self_attn.linear_out.", ".self_attn.o_proj.")
        nk = nk.replace(".self_attn.linear_pos.", ".self_attn.relative_k_proj.")
        nk = nk.replace(".self_attn.pos_bias_u", ".self_attn.bias_u")
        nk = nk.replace(".self_attn.pos_bias_v", ".self_attn.bias_v")
        return nk
    if k.startswith("transformer_encoder."):
        if k.endswith("first_sub_layer.key_net.bias"):
            return None  # this port's k_proj has no bias (softmax no-op)
        nk = k
        nk = nk.replace(".first_sub_layer.query_net.", ".self_attn.q_proj.")
        nk = nk.replace(".first_sub_layer.key_net.", ".self_attn.k_proj.")
        nk = nk.replace(".first_sub_layer.value_net.", ".self_attn.v_proj.")
        nk = nk.replace(".first_sub_layer.out_projection.", ".self_attn.out_proj.")
        nk = nk.replace(".layer_norm_1.", ".self_attn_layer_norm.")
        nk = nk.replace(".second_sub_layer.dense_in.", ".fc1.")
        nk = nk.replace(".second_sub_layer.dense_out.", ".fc2.")
        nk = nk.replace(".layer_norm_2.", ".final_layer_norm.")
        return nk
    if k.startswith("sortformer_modules."):
        return k
    return None


def convert_nemo(nemo_state_path, num_spk: int = 4):
    """Build the v2 streaming model and load the converted NeMo weights."""
    model = build_streaming_v2_model(num_spk=num_spk)
    nemo_sd = torch.load(nemo_state_path, map_location="cpu")
    target = model.state_dict()
    new_sd, dropped = {}, []
    for k, v in nemo_sd.items():
        nk = remap_key(k)
        if nk is None:
            dropped.append(k)
            continue
        if nk in target and tuple(target[nk].shape) == tuple(v.shape):
            new_sd[nk] = v
        else:
            dropped.append(f"{k}->{nk}?(shape)")
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    report = dict(
        n_nemo=len(nemo_sd),
        n_loaded=len(new_sd),
        missing=[
            m
            for m in missing
            if not m.startswith(("preprocessor.fb", "preprocessor.window"))
        ],
        unexpected=list(unexpected),
        dropped_examples=[d for d in dropped if "?" in d][:10],
    )
    return model, report
