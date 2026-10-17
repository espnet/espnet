import pytest


@pytest.fixture
def tiny_geolid(tmp_path):
    transformers = pytest.importorskip("transformers")
    from espnet2.tasks.lid import LIDTask

    config = transformers.Wav2Vec2Config(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=[8],
        conv_stride=[2],
        conv_kernel=[3],
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        layerdrop=0.0,
    )
    base = tmp_path / "base"
    transformers.Wav2Vec2Model(config).save_pretrained(base)
    args = LIDTask.get_parser().parse_args([])
    args.model = "upstream_condition"
    args.model_conf = dict(
        legacy_inference=False,
        lang2vec_conditioning_layers=[0, 1],
        shared_conditioning_proj=True,
        apply_intermediate_lang2vec_loss=True,
        inter_lang2vec_loss_weight=0.4,
        cutoff_gradient_from_backbone=False,
        cutoff_gradient_before_condproj=True,
    )
    args.frontend = "s3prl_condition"
    args.frontend_conf = dict(
        frontend_conf=dict(upstream="hf_wav2vec2_condition", path_or_url=str(base)),
        multilayer_feature=True,
    )
    args.encoder = "ecapa_tdnn"
    args.encoder_conf = dict(ndim=8, output_size=12, model_scale=2)
    args.pooling = "chn_attn_stat"
    args.projector = "rawnet3"
    args.projector_conf = dict(output_size=4)
    args.encoder_condition = "identity"
    args.pooling_condition = "chn_attn_stat"
    args.projector_condition = "rawnet3"
    args.projector_condition_conf = dict(output_size=4)
    args.normalize = "utterance_mvn"
    args.normalize_conf = dict(norm_vars=False)
    args.loss = "aamsoftmax_sc_topk_lang2vec"
    args.loss_conf = dict(
        lang2vec_dim=3, lang2vec_type="geo", lang2vec_weight=0.2, k_top=2
    )
    args.lang_num = 4
    return args, LIDTask.build_model(args), base
