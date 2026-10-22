import pytest
import torch


@pytest.mark.parametrize(
    "condition,intermediate", [(True, True), (False, True), (True, False)]
)
def test_training_and_inference(tiny_geolid, condition, intermediate):
    _, model, _ = tiny_geolid
    if not condition:
        for layer in model.conditioned_layers:
            layer.conditioning_layer = None
    model.apply_intermediate_lang2vec_loss = intermediate
    speech = torch.randn(2, 880)
    lengths = torch.tensor([880, 810])
    loss, stats, weight = model(
        speech, lengths, lid_labels=torch.tensor([0, 1]), lang2vecs=torch.randn(2, 3)
    )
    assert torch.isfinite(loss)
    auxiliary = stats[
        "lang2vec_loss_all" if intermediate else "lang2vec_loss_downstream"
    ]
    torch.testing.assert_close(
        loss.detach(), 0.8 * stats["class_loss"] + 0.2 * auxiliary
    )
    loss.backward()
    assert (
        model.conditioned_layers[0].prediction.weight.grad is not None
    ) == intermediate
    assert weight == 2
    model.eval()
    with torch.no_grad():
        embeddings, predictions = model(speech, lengths, extract_embd=True)
        assert embeddings.shape == (2, 4)
        assert predictions.shape == (2,)
        assert model.collect_feats(speech, lengths)["feats"].shape[:2] == (2, 3)
        assert len(model.frontend(speech, lengths)) == 2
        assert not any(k.startswith("upstream_encoder.") for k in model.state_dict())


@pytest.mark.parametrize("supervised", [True, False])
def test_masked_lang2vec_training(tiny_geolid, supervised, monkeypatch):
    _, model, _ = tiny_geolid
    predictions = []
    original = model.compute_lang2vec_loss

    def record(prediction, target, mask=None):
        prediction.retain_grad()
        predictions.append(prediction)
        return original(prediction, target, mask)

    monkeypatch.setattr(model, "compute_lang2vec_loss", record)
    targets = torch.full((2, 3), float("nan"))
    if supervised:
        targets[0] = torch.randn(3)
    loss, stats, _ = model(
        torch.randn(2, 1600),
        torch.tensor([1600, 980]),
        lid_labels=torch.tensor([0, 1]),
        lang2vecs=targets,
        lang2vec_mask=torch.tensor([[int(supervised)], [0]]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.loss.weight.grad is not None
    assert torch.isfinite(model.loss.weight.grad).all()
    if supervised:
        # Both intermediate heads and the downstream head ignore the masked row.
        assert len(predictions) == 1 + len(model.conditioned_layers)
        for prediction in predictions:
            assert prediction.grad[0].abs().sum() > 0
            assert torch.count_nonzero(prediction.grad[1]) == 0
    else:
        torch.testing.assert_close(loss.detach(), stats["class_loss"])
        assert not predictions
        assert model.lang2vec_head.weight.grad is None
        auxiliary_stats = [
            value for name, value in stats.items() if "lang2vec_loss" in name
        ]
        assert auxiliary_stats
        assert all(torch.isnan(value) for value in auxiliary_stats)


def test_all_masked_auxiliary_component(tiny_geolid):
    _, module, _ = tiny_geolid
    prediction = torch.randn(2, 3, requires_grad=True)
    loss = module.compute_lang2vec_loss(
        prediction, torch.full_like(prediction, float("nan")), torch.zeros(2, 1)
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == 0


def test_stock_frontend_initialization(tiny_geolid):
    from espnet2.asr.frontend.s3prl import S3prlFrontend
    from espnet2.torch_utils.initialize import initialize

    _, model, _ = tiny_geolid
    assert type(model.frontend) is S3prlFrontend
    assert model.frontend.upstream.upstream.model.config.layerdrop == 0
    original = (
        model.conditioned_layers[0].layer.attention.q_proj.weight.detach().clone()
    )
    initialize(model, "xavier_uniform")
    torch.testing.assert_close(
        model.conditioned_layers[0].layer.attention.q_proj.weight, original
    )
    assert (
        model.conditioned_layers[0].conditioning_layer
        is model.conditioned_layers[1].conditioning_layer
    )


def test_intermediate_predictions_do_not_leak(tiny_geolid):
    _, model, _ = tiny_geolid
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _, _, predictions = model.extract_feats(
                torch.randn(2, 1600),
                torch.tensor([1600, 980]),
                return_intermediate=True,
            )
            assert len(predictions) == 2
            assert all(p.shape == (2, 3) for p in predictions)
            assert all(
                layer.intermediate_out is None for layer in model.conditioned_layers
            )


@pytest.mark.parametrize(
    "tiny_geolid",
    [
        {"shared_conditioning_proj": shared, "cutoff_gradient_from_backbone": frozen}
        for shared in (True, False)
        for frozen in (True, False)
    ],
    indirect=True,
)
def test_projection_sharing_and_gradients(tiny_geolid):
    args, model, _ = tiny_geolid
    first, second = model.conditioned_layers
    shared = args.model_conf["shared_conditioning_proj"]
    frozen = args.model_conf["cutoff_gradient_from_backbone"]
    assert (first.conditioning_layer is second.conditioning_layer) == shared
    loss, _, _ = model(
        torch.randn(2, 1600),
        torch.tensor([1600, 980]),
        lid_labels=torch.tensor([0, 1]),
        lang2vecs=torch.randn(2, 3),
    )
    loss.backward()
    assert (first.conditioning_layer.weight.grad is None) == frozen
    assert first.prediction.weight.grad is not None
    assert first.layer.attention.q_proj.weight.grad is not None


def test_stock_gradient_checkpointing(tiny_geolid):
    _, model, _ = tiny_geolid
    model.train()
    model.frontend.upstream.upstream.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    loss, _, _ = model(
        torch.randn(2, 1600),
        torch.tensor([1600, 980]),
        lid_labels=torch.tensor([0, 1]),
        lang2vecs=torch.randn(2, 3),
    )
    loss.backward()
    assert model.conditioned_layers[0].prediction.weight.grad is not None


def test_default_model_classification_contract(tiny_geolid):
    from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
        ArcMarginProduct_intertopk_subcenter,
    )
    from espnet2.tasks.lid import LIDTask

    args, conditioned, _ = tiny_geolid
    assert type(conditioned.loss) is ArcMarginProduct_intertopk_subcenter
    args.model_conf = {}
    model = LIDTask.build_model(args)
    assert model.frontend.upstream.upstream.model.config.layerdrop == 0.8
    assert model.conditioned_layers == []
    assert model.lang2vec_head is None
    loss, _, _ = model(
        torch.randn(2, 1600),
        torch.tensor([1600, 980]),
        lid_labels=torch.tensor([0, 1]),
    )
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize(
    "tiny_geolid", [{"lang2vec_loss": "mse"}, {"lang2vec_loss": "bce"}], indirect=True
)
def test_auxiliary_loss_and_gradient(tiny_geolid):
    args, model, _ = tiny_geolid
    inputs = torch.randn(2, 4, requires_grad=True)
    targets = torch.rand(2, 3)
    prediction = model.lang2vec_head(inputs)
    actual = model.compute_lang2vec_loss(prediction, targets)
    criterion = (
        torch.nn.functional.mse_loss
        if args.model_conf["lang2vec_loss"] == "mse"
        else torch.nn.functional.binary_cross_entropy_with_logits
    )
    torch.testing.assert_close(actual, criterion(prediction, targets))
    actual.backward()
    assert model.lang2vec_head.weight.grad.abs().sum() > 0
    assert inputs.grad.abs().sum() > 0


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_attention_backends_preserve_valid_lengths(tiny_geolid, implementation):
    _, model, _ = tiny_geolid
    upstream = model.frontend.upstream.upstream.model
    upstream.set_attn_implementation(implementation)
    model.eval()
    lengths_seen = []
    handles = [
        layer.pooling.register_forward_pre_hook(
            lambda module, args, kwargs: lengths_seen.append(
                kwargs["feat_lengths"].clone()
            ),
            with_kwargs=True,
        )
        for layer in model.conditioned_layers
    ]
    try:
        with torch.no_grad():
            embeddings, predictions = model(
                torch.randn(2, 1600), torch.tensor([1600, 980]), extract_embd=True
            )
        assert torch.isfinite(embeddings).all()
        assert predictions.shape == (2,)
        # Valid convolution output lengths, independently of the attention mask.
        assert len(lengths_seen) == 2
        for lengths in lengths_seen:
            torch.testing.assert_close(lengths, torch.tensor([4, 2]))
    finally:
        for handle in handles:
            handle.remove()


def test_conditioned_checkpoint_keys_and_strict_reload(tiny_geolid):
    from espnet2.lid.espnet_model import ESPnetLIDModel
    from espnet2.tasks.lid import LIDTask

    args, model, _ = tiny_geolid
    assert type(model) is ESPnetLIDModel
    state = model.state_dict()
    assert state["lang2vec_head.weight"].shape == (3, 4)
    assert state["lang2vec_head.bias"].shape == (3,)
    prefixes = [
        f"frontend.upstream.upstream.model.encoder.layers.{index}." for index in (0, 1)
    ]
    for prefix in prefixes:
        # Preserve published wrapper names, including both aliases of a shared
        # conditioning projection; strict loading must not need a tensor rename.
        assert state[prefix + "layer.attention.q_proj.weight"].shape == (8, 8)
        assert state[prefix + "prediction.weight"].shape == (3, 4)
        assert state[prefix + "conditioning_layer.weight"].shape == (8, 3)
    assert (
        state[prefixes[0] + "conditioning_layer.weight"].data_ptr()
        == state[prefixes[1] + "conditioning_layer.weight"].data_ptr()
    )
    restored = LIDTask.build_model(args)
    restored.load_state_dict(state, strict=True)
    assert set(restored.state_dict()) == set(state)
    model.eval()
    restored.eval()
    speech, lengths = torch.randn(2, 1600), torch.tensor([1600, 980])
    with torch.no_grad():
        expected = model(speech, lengths, extract_embd=True)
        actual = restored(speech, lengths, extract_embd=True)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
