import pytest
import torch

from espnet2.asr.frontend.huggingface import HuggingFaceFrontend

pytest.importorskip("transformers", minversion="4.43.0")


@pytest.mark.parametrize(
    "model, fs",
    [
        ("taiqihe/test-w2v-bert-dummy", 16000),
    ],
)
@pytest.mark.execution_timeout(10)
def test_frontend_backward(model, fs):
    frontend = HuggingFaceFrontend(
        model,
        fs=fs,
        download_dir="./hf_cache",
        load_pretrained=False,
    )
    test_length = 640
    wavs = torch.randn(2, test_length, requires_grad=True)
    lengths = torch.LongTensor([test_length, test_length])
    feats, f_lengths = frontend(wavs, lengths)
    feats.sum().backward()


@pytest.mark.parametrize("feat_extract_norm", ["group", "layer"])
@pytest.mark.execution_timeout(10)
def test_frontend_lengths_of_waveform_model(tmp_path, feat_extract_norm):
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Model,
    )

    # The layer-norm models, like wav2vec2-xls-r, come with an attention mask, the
    # group-norm ones, like wav2vec2-base, without.
    Wav2Vec2Model(
        Wav2Vec2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8, 8, 8, 8, 8, 8),
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            feat_extract_norm=feat_extract_norm,
        )
    ).save_pretrained(tmp_path)
    Wav2Vec2FeatureExtractor(
        sampling_rate=16000, return_attention_mask=feat_extract_norm == "layer"
    ).save_pretrained(tmp_path)

    frontend = HuggingFaceFrontend(str(tmp_path), fs=16000).eval()
    wavs = torch.randn(3, 8000)
    lengths = torch.LongTensor([8000, 6000, 4000])
    feats, f_lengths = frontend(wavs, lengths)
    assert f_lengths.shape == (3,)
    assert f_lengths.max() == feats.size(1)
    for i in range(3):
        alone, alone_lengths = frontend(
            wavs[i : i + 1, : lengths[i]], lengths[i : i + 1]
        )
        assert alone_lengths.tolist() == [alone.size(1)]
        assert f_lengths[i].item() == alone.size(1)
