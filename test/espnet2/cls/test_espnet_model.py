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
