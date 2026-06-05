"""Initialize the Sortformer FastConformer encoder from NVIDIA NEST weights.

NVIDIA trains Sortformer by initializing its FastConformer encoder from the
self-supervised **NEST** model (``nvidia/ssl_en_nest_large_v1.0``) and training
the Transformer + diarization head from scratch on diarization data. NEST-L's
FastConformer is architecturally identical to Sortformer's encoder (18 layers,
d_model 512, ``dw_striding`` 8x, conv kernel 9), so its weights map 1:1 into this
port with a NeMo-name -> port-name remap.

Export the NEST encoder weights once (in the NeMo env)::

    from nemo.collections.asr.models import ASRModel
    m = ASRModel.from_pretrained("nvidia/ssl_en_nest_large_v1.0", map_location="cpu")
    import torch; torch.save(m.encoder.state_dict(), "nest_encoder.pt")

then call :func:`load_nest_encoder` to initialize a Sortformer model's encoder.
"""

import torch


def nest_to_encoder_state_dict(nest_sd):
    """Remap a NeMo ConformerEncoder state-dict to this FastConformerEncoder."""
    out = {}
    for k, v in nest_sd.items():
        nk = k
        # subsampling
        nk = nk.replace("pre_encode.conv.", "subsampling.layers.")
        nk = nk.replace("pre_encode.out.", "subsampling.linear.")
        # conv module batch norm
        nk = nk.replace(".conv.batch_norm.", ".conv.norm.")
        # self-attention projections
        nk = nk.replace(".self_attn.linear_q.", ".self_attn.q_proj.")
        nk = nk.replace(".self_attn.linear_k.", ".self_attn.k_proj.")
        nk = nk.replace(".self_attn.linear_v.", ".self_attn.v_proj.")
        nk = nk.replace(".self_attn.linear_out.", ".self_attn.o_proj.")
        nk = nk.replace(".self_attn.linear_pos.", ".self_attn.relative_k_proj.")
        nk = nk.replace(".self_attn.pos_bias_u", ".self_attn.bias_u")
        nk = nk.replace(".self_attn.pos_bias_v", ".self_attn.bias_v")
        out[nk] = v
    return out


def load_nest_encoder(model, nest_encoder_path, verbose=True):
    """Load NEST weights into ``model.encoder`` in place. Returns a report dict."""
    nest_sd = torch.load(nest_encoder_path, map_location="cpu")
    enc_sd = nest_to_encoder_state_dict(nest_sd)
    target = model.encoder.state_dict()
    matched = {
        k: v
        for k, v in enc_sd.items()
        if k in target and tuple(target[k].shape) == tuple(v.shape)
    }
    missing, unexpected = model.encoder.load_state_dict(matched, strict=False)
    report = dict(
        n_nest=len(nest_sd),
        n_loaded=len(matched),
        missing=list(missing),
        unexpected=[k for k in enc_sd if k not in target],
    )
    if verbose:
        print(
            f"NEST init: loaded {report['n_loaded']}/{len(target)} encoder tensors "
            f"({report['n_nest']} in NEST); unexpected={report['unexpected'][:5]}"
        )
    return report
