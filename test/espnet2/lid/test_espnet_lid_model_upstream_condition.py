import pytest
import torch

from espnet2.lid.espnet_model_upstream_condition import ESPnetLIDUpstreamConditionModel
from espnet2.lid.loss.aamsoftmax_sc_topk_lang2vec import AAMSoftmaxSCTopKLang2Vec


def test_import_espnet_lid_upstream_condition_model():
    """Test that ESPnetLIDUpstreamConditionModel can be imported."""
    assert ESPnetLIDUpstreamConditionModel is not None


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    "lang2vec_type", ["geo", "phonology_knn", "syntax_knn", "inventory_knn"]
)
def test_aamsoftmax_sc_topk_lang2vec_loss(training, lang2vec_type):

    inputs = torch.randn(2, 198, requires_grad=True)
    lid_labels = torch.randint(0, 10, (2,)).long()
    if lang2vec_type == "geo":
        lang2vecs = torch.randn(2, 299)
        lang2vec_dim = 299  # mock data
    else:
        lang2vecs = torch.randn(2, 28)
        lang2vec_dim = 28  # mock data

    aamsoftmax_sc_topk_lang2vec_loss = AAMSoftmaxSCTopKLang2Vec(
        nout=198,  # Use fixed embedding dimension
        nclasses=10,
        scale=15,
        margin=0.3,
        easy_margin=False,
        K=3,
        mp=0.06,
        k_top=5,
        do_lm=False,
        lang2vec_dim=lang2vec_dim,
        lang2vec_type=lang2vec_type,
        lang2vec_weight=0.2,
    )

    kwargs = {
        "input": inputs,
        "label": lid_labels,
        "lang2vec": lang2vecs,
    }

    if training:
        loss, *_ = aamsoftmax_sc_topk_lang2vec_loss(**kwargs)
        loss.backward()
    else:
        loss, *_ = aamsoftmax_sc_topk_lang2vec_loss(**kwargs)


@pytest.mark.parametrize(
    "condition,intermediate", [(True, True), (False, True), (True, False)]
)
def test_training_and_inference(tiny_geolid, condition, intermediate):
    _, model, _ = tiny_geolid
    model.upstream_encoder.apply_intermediate_lang2vec_condition = condition
    model.apply_intermediate_lang2vec_loss = intermediate
    speech = torch.randn(2, 880)
    lengths = torch.tensor([880, 810])
    loss, stats, weight = model(
        speech, lengths, lid_labels=torch.tensor([0, 1]), lang2vecs=torch.randn(2, 3)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert (
        model.upstream_encoder.lang2vec_head["0"][0].weight.grad is not None
    ) == intermediate
    assert weight == 2
    model.eval()
    with torch.no_grad():
        embeddings, predictions = model(speech, lengths, extract_embd=True)
        assert embeddings.shape == (2, 4)
        assert predictions.shape == (2,)
        assert model.collect_feats(speech, lengths)["feats"].shape[:2] == (2, 440)
        assert len(model.frontend(speech, lengths)) == 2
        assert not any(k.startswith("upstream_encoder.") for k in model.state_dict())


@pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
def test_loss_components_are_not_mutated(weight):
    module = AAMSoftmaxSCTopKLang2Vec(
        nout=4,
        nclasses=4,
        k_top=2,
        lang2vec_dim=3,
        lang2vec_type="geo",
        lang2vec_weight=weight,
    )
    x, labels, targets = torch.randn(2, 4), torch.tensor([0, 1]), torch.randn(2, 3)
    expected = module(x, labels)[0]
    total, _, _, classification, auxiliary = module(x, labels, targets)
    torch.testing.assert_close(classification, expected)
    torch.testing.assert_close(total, (1 - weight) * expected + weight * auxiliary)
