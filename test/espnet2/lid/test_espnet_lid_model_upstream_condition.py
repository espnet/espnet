import pytest

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.lid.espnet_model_upstream_condition import ESPnetLIDUpstreamConditionModel
from espnet2.lid.frontend.s3prl_condition import S3prlFrontendCondition
from espnet2.lid.loss.aamsoftmax_sc_topk_lang2vec import AAMSoftmaxSCTopKLang2Vec
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector

default_frontend = DefaultFrontend(
    fs=16000,
    n_mels=80,
)

specaug = SpecAug(
    apply_time_warp=True,
    time_warp_window=5,
    time_warp_mode="bicubic",
    apply_freq_mask=True,
    freq_mask_width_range=[0, 30],
    num_freq_mask=2,
    apply_time_mask=True,
    time_mask_width_range=[0, 40],
    num_time_mask=2,
)

normalize = UtteranceMVN()

ecapa_tdnn_encoder = EcapaTdnnEncoder(
    default_frontend.output_size(), model_scale=2, ndim=16, output_size=24
)

chan_attn_stat_pooling = ChnAttnStatPooling(input_size=ecapa_tdnn_encoder.output_size())

rawnet3_projector = RawNet3Projector(
    input_size=chan_attn_stat_pooling.output_size(), output_size=8
)

aamsoftmax_sc_topk_lang2vec_loss = AAMSoftmaxSCTopKLang2Vec(
    nout=rawnet3_projector.output_size(),
    nclasses=10,
    scale=15,
    margin=0.3,
    easy_margin=False,
    K=3,
    mp=0.06,
    k_top=5,
    do_lm=False,
    lang2vec_dim=299,
    lang2vec_type="geo",
    lang2vec_weight=0.2,
)


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
def test_aamsoftmax_sc_topk_lang2vec_loss(training):
    inputs = torch.randn(2, 8000)
    ilens = torch.LongTensor([8000, 7800])
    lid_labels = torch.randint(0, 10, (2,))
    lid_model = ESPnetLIDModel(
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        encoder=encoder,
        pooling=pooling,
        projector=projector,
        loss=aamsoftmax_sc_topk_lang2vec_loss,
    )
    lang2vecs = torch.randn(2, 299)

    if training:
        lid_model.train()
    else:
        lid_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "lid_labels": lid_labels,
        "lang2vecs": lang2vecs,
    }

    if training:
        loss, *_ = lid_model(**kwargs)
        loss.backward()
    else:
        loss, *_ = lid_model(**kwargs)
