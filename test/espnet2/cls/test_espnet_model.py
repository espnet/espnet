import numpy as np
import pytest
import torch
from scipy import stats
from sklearn import metrics

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


def calculate_stats_testing_internal_(
    output, target, classification_type="multi-label"
):
    """Calculate statistics including mAP, AUC, etc.
    This function is adapted from the official implementation of AST
    https://github.com/YuanGongND/ast/blob/master/src/utilities/stats.py

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      classification_type: str, 'multi-label' or 'multi-class'.
        Newly introduced parameter.

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    if classification_type == "multi-class":
        return [{"acc": acc}]

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None
        )

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k]
        )

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000  # Sample statistics to reduce size
        dict = {
            "precisions": precisions[0::save_every_steps],
            "recalls": recalls[0::save_every_steps],
            "AP": avg_precision,
            "fpr": fpr[0::save_every_steps],
            "fnr": 1.0 - tpr[0::save_every_steps],
            "auc": auc,
            # Acc is not class wise, here for consistency
            "acc": acc,
        }
        stats.append(dict)

    return stats


def test_metrics_multilabel():
    token_list = [
        "class0",
        "class1",
        "class2",
        "class3",
        "class4",
        "class5",
        "class6",
        "<unk>",
    ]
    n_classes = len(token_list) - 1
    enc_out = 4
    encoder = TransformerEncoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
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
        classification_type="multi-label",
    )

    batch_size = 55
    preds = torch.randn(batch_size, n_classes)
    target = torch.randint(0, 2, (batch_size, n_classes))
    # The additional tensors ensure that all class values are
    # present in the targets.
    # This is necessary to avoid a ZeroDivisionError
    # in the official implementation of AST.
    ones = torch.ones(1, n_classes)
    zeros = torch.zeros(1, n_classes)
    preds = torch.cat([preds, zeros, ones], dim=0)
    target = torch.cat([target, zeros, ones], dim=0)
    batch_size += 2

    stats_espnet_cls = {}
    for metric_name, metric_fn in model.metric_functions.items():
        val = metric_fn(preds, target).detach()
        stats_espnet_cls[metric_name] = val
    stats_official_ast = calculate_stats_testing_internal_(
        preds.numpy(), target.numpy()
    )
    mAP = np.mean([stat["AP"] for stat in stats_official_ast])
    assert np.isclose(mAP, stats_espnet_cls["mAP"].item(), atol=1e-4)


def test_metrics_multiclass():
    token_list = [
        "class0",
        "class1",
        "class2",
        "class3",
        "class4",
        "class5",
        "class6",
        "<unk>",
    ]
    n_classes = len(token_list) - 1
    enc_out = 4
    encoder = TransformerEncoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
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
        classification_type="multi-class",
    )

    batch_size = 55
    preds = torch.randn(batch_size, n_classes)
    target = torch.randint(0, n_classes, (batch_size, 1))

    stats_espnet_cls = {}
    for metric_name, metric_fn in model.metric_functions.items():
        val = metric_fn(preds, target.squeeze(-1)).detach()
        stats_espnet_cls[metric_name] = val
    onehot_tgt = label_to_onehot(
        target, torch.ones_like(target), n_classes, "multi-class"
    )
    stats_official_ast = calculate_stats_testing_internal_(
        preds.numpy(), onehot_tgt.numpy(), classification_type="multi-class"
    )
    acc = np.mean([stat["acc"] for stat in stats_official_ast])
    assert np.isclose(acc, stats_espnet_cls["acc"].item(), atol=1e-4)


@pytest.mark.parametrize("batch_size", [40, 1000, 25000])
def test_metrics_mAP_over_multiple_batches(batch_size):
    token_list = [
        "class0",
        "class1",
        "class2",
        "class3",
        "class4",
        "class5",
        "class6",
        "<unk>",
    ]
    n_classes = len(token_list) - 1
    enc_out = 4
    encoder = TransformerEncoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
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
        classification_type="multi-label",
        log_epoch_metrics=True,
    )

    batch_size = 55
    preds = torch.randn(batch_size, n_classes)
    target = torch.randint(0, 2, (batch_size, n_classes))
    # The additional tensors ensure that all class values are
    # present in the targets.
    # This is necessary to avoid a ZeroDivisionError
    # in the official implementation of AST.
    ones = torch.ones(1, n_classes)
    zeros = torch.zeros(1, n_classes)
    preds = torch.cat([preds, zeros, ones], dim=0)
    target = torch.cat([target, zeros, ones], dim=0)
    batch_size += 2

    model.eval()  # validation mode and split
    model.metric_functions["mAP"](
        preds[: batch_size // 2],
        target[: batch_size // 2],
    )
    model.metric_functions["mAP"](
        preds[batch_size // 2 :],
        target[batch_size // 2 :],
    )
    # validation mode, accumulated
    mAP_split = model.validation_epoch_end_()["epoch_mAP"]

    model.train()  # train mode and bulk
    mAP_bulk = model.metric_functions["mAP"](
        preds,
        target,
    ).item()
    model.training_epoch_end_()

    stats_official_ast = calculate_stats_testing_internal_(
        preds.numpy(), target.numpy()
    )
    mAP_official = np.mean([stat["AP"] for stat in stats_official_ast])
    assert np.isclose(mAP_official, mAP_split, atol=1e-4)
    assert np.isclose(mAP_split, mAP_bulk, atol=1e-4)


class EncoderCombiner:
    def combine_encodings(
        self,
        text_encoding: torch.Tensor,
        text_encoding_lens: torch.Tensor,
        speech_encoding: torch.Tensor,
        speech_encoding_lens: torch.Tensor,
    ):
        batch_size, _, dim = text_encoding.shape
        encoder_out_lens = text_encoding_lens + speech_encoding_lens
        max_len = encoder_out_lens.max()

        encoder_out = torch.zeros(
            (batch_size, max_len, dim),
            dtype=text_encoding.dtype,
            device=text_encoding.device,
        )

        for i in range(batch_size):
            text_len = text_encoding_lens[i].item()
            speech_len = speech_encoding_lens[i].item()

            encoder_out[i, :text_len] = text_encoding[i, :text_len]
            encoder_out[i, text_len : text_len + speech_len] = speech_encoding[
                i, :speech_len
            ]

        return encoder_out, encoder_out_lens


@pytest.fixture
def encoder_combiner():
    return EncoderCombiner()


def test_basic_combination(encoder_combiner):
    text_encoding = torch.tensor(
        [
            [[1, 1], [2, 2], [0, 0]],  # Padded text encoding (Batch 1)
            [[3, 3], [0, 0], [0, 0]],  # Padded text encoding (Batch 2)
        ],
        dtype=torch.float32,
    )

    text_encoding_lens = torch.tensor([2, 1])  # Actual lengths

    speech_encoding = torch.tensor(
        [
            [[4, 4], [5, 5]],  # Speech encoding (Batch 1)
            [[6, 6], [7, 7]],  # Speech encoding (Batch 2)
        ],
        dtype=torch.float32,
    )

    speech_encoding_lens = torch.tensor([2, 2])  # Actual lengths

    expected_output = torch.tensor(
        [
            [[1, 1], [2, 2], [4, 4], [5, 5]],  # Combined (Batch 1)
            [[3, 3], [6, 6], [7, 7], [0, 0]],  # Combined (Batch 2, padded)
        ],
        dtype=torch.float32,
    )

    expected_lens = torch.tensor([4, 3])

    encoder_out, encoder_out_lens = encoder_combiner.combine_encodings(
        text_encoding, text_encoding_lens, speech_encoding, speech_encoding_lens
    )

    assert torch.equal(encoder_out, expected_output)
    assert torch.equal(encoder_out_lens, expected_lens)


def test_empty_text_encoding(encoder_combiner):
    """Tests when text encoding is empty (only speech)."""
    text_encoding = torch.zeros((2, 0, 2))  # No text
    text_encoding_lens = torch.tensor([0, 0])

    speech_encoding = torch.tensor(
        [[[4, 4], [5, 5]], [[6, 6], [7, 7]]], dtype=torch.float32
    )

    speech_encoding_lens = torch.tensor([2, 2])

    expected_output = speech_encoding
    expected_lens = speech_encoding_lens

    encoder_out, encoder_out_lens = encoder_combiner.combine_encodings(
        text_encoding, text_encoding_lens, speech_encoding, speech_encoding_lens
    )

    assert torch.equal(encoder_out, expected_output)
    assert torch.equal(encoder_out_lens, expected_lens)


def test_empty_speech_encoding(encoder_combiner):
    """Tests when speech encoding is empty (only text)."""
    text_encoding = torch.tensor(
        [[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=torch.float32
    )

    text_encoding_lens = torch.tensor([2, 2])

    speech_encoding = torch.zeros((2, 0, 2))  # No speech
    speech_encoding_lens = torch.tensor([0, 0])

    expected_output = text_encoding
    expected_lens = text_encoding_lens

    encoder_out, encoder_out_lens = encoder_combiner.combine_encodings(
        text_encoding, text_encoding_lens, speech_encoding, speech_encoding_lens
    )

    assert torch.equal(encoder_out, expected_output)
    assert torch.equal(encoder_out_lens, expected_lens)


def test_varying_lengths(encoder_combiner):
    """Tests when text and speech have very different lengths."""
    text_encoding = torch.tensor(
        [[[1, 1]], [[2, 2], [3, 3], [4, 4]]],  # Only one token  # Three tokens
        dtype=torch.float32,
    )

    text_encoding_lens = torch.tensor([1, 3])

    speech_encoding = torch.tensor(
        [[[5, 5], [6, 6], [7, 7]], [[8, 8]]],  # Three tokens  # Only one token
        dtype=torch.float32,
    )

    speech_encoding_lens = torch.tensor([3, 1])

    expected_output = torch.tensor(
        [
            [[1, 1], [5, 5], [6, 6], [7, 7]],  # Combined (Batch 1)
            [[2, 2], [3, 3], [4, 4], [8, 8]],  # Combined (Batch 2)
        ],
        dtype=torch.float32,
    )

    expected_lens = torch.tensor([4, 4])

    encoder_out, encoder_out_lens = encoder_combiner.combine_encodings(
        text_encoding, text_encoding_lens, speech_encoding, speech_encoding_lens
    )

    assert torch.equal(encoder_out, expected_output)
    assert torch.equal(encoder_out_lens, expected_lens)
