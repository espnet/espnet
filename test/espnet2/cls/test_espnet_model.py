import pytest
import torch


from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.cls.decoder.linear_decoder import LinearDecoder
from espnet2.cls.espnet_model import ESPnetClassificationModel, label_to_onehot


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, ConformerEncoder])
@pytest.mark.parametrize("classification_type", ["multi-label", "multi-class"])
def test_fwd_bkwd(encoder_arch, classification_type):
    token_list = ["class0", "class1", "class2", "class3", "class4", "<unk>"]
    n_classes = len(token_list) - 1
    enc_out = 4
    encoder = encoder_arch(20, output_size=enc_out, linear_units=4, num_blocks=2)
    decoder = LinearDecoder(n_classes, enc_out, pooling="mean")

    model = ESPnetClassificationModel(
        n_classes,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        decoder=decoder,
        classification_type=classification_type,
    )

    if classification_type == "multi-label":
        inputs = dict(
            speech=torch.randn(2, 10, 20, requires_grad=True),
            speech_lengths=torch.tensor([10, 8], dtype=torch.long),
            label=torch.tensor([[3, 1], [0, 4]], dtype=torch.long),
            label_lengths=torch.tensor([2, 2], dtype=torch.long),
        )
    else:
        inputs = dict(
            speech=torch.randn(2, 10, 20, requires_grad=True),
            speech_lengths=torch.tensor([10, 8], dtype=torch.long),
            label=torch.tensor([[3], [0]], dtype=torch.long),
            label_lengths=torch.tensor([[1], [1]], dtype=torch.long),
        )
    loss, _, _ = model(**inputs)
    loss.backward()


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, ConformerEncoder])
@pytest.mark.parametrize("classification_type", ["multi-label", "multi-class"])
def test_score(encoder_arch, classification_type):
    token_list = ["class0", "class1", "class2", "class3", "class4", "<unk>"]
    n_classes = len(token_list) - 1
    enc_out = 4
    encoder = encoder_arch(20, output_size=enc_out, linear_units=4, num_blocks=2)
    decoder = LinearDecoder(n_classes, enc_out, pooling="mean")

    model = ESPnetClassificationModel(
        n_classes,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        decoder=decoder,
        classification_type=classification_type,
    )
    inputs = dict(
        speech=torch.randn(1, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10], dtype=torch.long),
    )
    model.eval()
    scores = model.score(**inputs)
    assert scores.shape == (1, n_classes)


@pytest.mark.parametrize(
    "label, label_lengths, vocab_size, classification_type, expected_onehot",
    [
        (
            torch.tensor([[1, 2, -1], [3, 1, 4], [1, -1, -1]], dtype=torch.long),
            torch.tensor([2, 3, 1], dtype=torch.long),
            5,
            "multi-label",
            torch.tensor(
                [
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1],
                    [0, 1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor([[1], [3], [2], [4]], dtype=torch.long),
            torch.tensor([1, 1, 1, 1], dtype=torch.long),
            5,
            "multi-class",
            torch.tensor(
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=torch.long,
            ),
        ),
    ],
)
def test_label_to_onehot(
    label, label_lengths, vocab_size, classification_type, expected_onehot
):
    onehot = label_to_onehot(label, label_lengths, vocab_size, classification_type)
    assert torch.all(onehot == expected_onehot)
