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
    """Remap a NeMo ConformerEncoder state-dict onto this FastConformerEncoder.

    Renames subsampling, conv-module batch-norm and self-attention projection
    keys from NeMo's naming to this port's. Keys are not filtered here; shape
    validation happens in :func:`load_nest_encoder`.

    Args:
        nest_sd: The NEST encoder state dict (from ``m.encoder.state_dict()``).

    Returns:
        A new state dict keyed with this FastConformerEncoder's names.
    """
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
    """Initialize a Sortformer model's encoder from exported NEST weights.

    Loads ONLY the FastConformer encoder weights (for initialization); the
    Transformer and diarization head are left untouched. The NEST encoder state
    dict must be exported beforehand in a NeMo environment (see the module
    docstring). Only shape-matching tensors are loaded, non-strictly and in
    place, into ``model.encoder``.

    Args:
        model: A Sortformer model exposing a ``.encoder`` submodule to fill.
        nest_encoder_path: Path to the exported NEST encoder state dict (a
            ``.pt`` saved from ``m.encoder.state_dict()``).
        verbose: If True, print a one-line load summary.

    Returns:
        A report dict with ``n_nest`` (tensors in the NEST file), ``n_loaded``
        (tensors transferred), ``missing`` (encoder keys left untouched) and
        ``unexpected`` (NEST keys with no matching target).

    Example:
        >>> from espnet3.systems.diar.sortformer.convert_nest import load_nest_encoder
        >>> report = load_nest_encoder(model, "nest_encoder.pt")
        >>> report["n_loaded"] > 0
        True
    """
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
