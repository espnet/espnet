from test.espnet2.cls.test_espnet_model import calculate_stats_testing_internal_

import numpy as np
import pytest
import torch

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.cls.decoder.linear_decoder import LinearDecoder
from espnet2.cls.espnet_model import ESPnetClassificationModel
from espnet2.cls.lightning_callbacks import MultilabelAUPRCCallback


class DummyLightningModule:
    def __init__(self):
        self.model = None

    def log(self, *args, **kwargs):
        pass


def test_metrics_multilabel():
    mAP_callback = MultilabelAUPRCCallback()
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
    lightning_module = DummyLightningModule()
    lightning_module.model = model

    mAP_callback.on_validation_start(None, lightning_module)
    mAP_callback.on_validation_epoch_start(None, lightning_module)

    batch_size = 10
    speech = torch.randn(batch_size, 500, 20)
    speech_lengths = torch.randint(1, 500, (batch_size,))
    label = torch.arange(n_classes).unsqueeze(0).expand(batch_size - 1, -1)
    label = torch.cat([label, torch.zeros(1, n_classes) - 1], dim=0).to(torch.long)
    # gt must have all labels
    label_lengths = torch.full((batch_size,), n_classes)
    model(speech, speech_lengths, label, label_lengths)

    preds = model.predictions[0]
    target = model.targets[0]

    mAP_callback.on_validation_batch_end(None, lightning_module, None, None, None)
    callback_mAP_value = mAP_callback.compute_mAP(None)
    stats_official_ast = calculate_stats_testing_internal_(
        preds.numpy(), target.numpy()
    )
    mAP = np.mean([stat["AP"] for stat in stats_official_ast])
    assert np.isclose(mAP, callback_mAP_value, atol=1e-4)
