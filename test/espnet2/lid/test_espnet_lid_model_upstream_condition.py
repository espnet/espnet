import pytest
import torch

from espnet2.lid.espnet_model_upstream_condition import ESPnetLIDUpstreamConditionModel
from espnet2.lid.frontend.s3prl_condition import S3prlFrontendCondition
from espnet2.lid.loss.aamsoftmax_sc_topk_lang2vec import AAMSoftmaxSCTopKLang2Vec


def test_import_espnet_lid_upstream_condition_model():
    """Test that ESPnetLIDUpstreamConditionModel can be imported."""
    assert ESPnetLIDUpstreamConditionModel is not None


def test_s3prl_frontend_condition_import_error():
    """Test S3prlFrontendCondition raises ImportError without modified S3PRL.

    This test is considered successful when ImportError is raised, indicating
    the class correctly detects missing modified S3PRL dependencies.
    """
    try:
        # This should raise ImportError due to missing modified s3prl
        S3prlFrontendCondition(
            fs=16000,
            frontend_conf=dict(
                upstream="hf_wav2vec2_condition",
                path_or_url="facebook/mms-1b",
            ),
        )
        pytest.fail("Expected ImportError was not raised")
    except ImportError as e:
        # Expected behavior - ImportError should be raised
        assert "s3prl is not found" in str(e)


def test_lid_model_upstream_condition_skip():
    """Placeholder test for ESPnetLIDUpstreamConditionModel

    Skipped since to use the ESPnetLIDUpstreamConditionModel, users need to install
    the modified S3PRL and Transformers by @Qingzheng-Wang.
    """
    pytest.skip(
        "ESPnetLIDUpstreamConditionModel requires modified S3PRL/Transformers "
        "by @Qingzheng-Wang."
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("lang2vec_type", ["geo", "phonology_knn", "syntax_knn", "inventory_knn"])
def test_aamsoftmax_sc_topk_lang2vec_loss(training, lang2vec_type):

    inputs = torch.randn(2, 198, requires_grad=True)
    lid_labels = torch.randint(0, 10, (2,)).long()
    if lang2vec_type == "geo":
        lang2vecs = torch.randn(2, 299)
        lang2vec_dim = 299 # mock data
    else:
        lang2vecs = torch.randn(2, 28)
        lang2vec_dim = 28 # mock data

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
