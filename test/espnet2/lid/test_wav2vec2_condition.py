import pytest
import torch

transformers = pytest.importorskip("transformers")


def test_no_mask_and_checkpointed_backward(tiny_geolid):
    _, model, _ = tiny_geolid
    upstream = model.frontend.upstream.upstream.model
    upstream.gradient_checkpointing_enable()
    model.train()
    outputs = upstream(torch.randn(2, 880), output_hidden_states=True)
    assert len(outputs.hidden_states) == 3
    assert len(outputs.intermediate_lang2vec_preds) == 2
    sum(x.square().mean() for x in outputs.intermediate_lang2vec_preds).backward()
    assert upstream.encoder.lang2vec_head["0"][0].weight.grad is not None


def test_old_projection_names_and_loss_alias(tiny_geolid):
    from espnet2.tasks.lid import LIDTask

    args, _, _ = tiny_geolid
    args.model_conf["shared_conditioning_proj"] = False
    model = LIDTask.build_model(args)
    state = model.state_dict()
    old = {}
    for k, v in state.items():
        if ".aamsoftmax_loss." in k:
            continue
        if ".lang2vec_conditioning_projs." in k:
            k = k.replace(".lang2vec_conditioning_projs.", ".conditioning_projs.")
            k = k.replace(".0.weight", ".weight").replace(".0.bias", ".bias")
        old[k] = v
    model.load_state_dict(old, strict=True)
