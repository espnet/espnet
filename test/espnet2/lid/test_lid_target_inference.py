"""Tests for LID training and inference with hard, soft and multi-label targets."""

import numpy as np
import pytest
import torch

from espnet2.lid.espnet_model import ESPnetLIDModel
from espnet2.spk.encoder.identity_encoder import IdentityEncoder
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
    ArcMarginProduct_intertopk_subcenter_multilabel,
    ArcMarginProduct_intertopk_subcenter_softtarget,
)
from espnet2.spk.pooling.mean_pooling import MeanPooling
from espnet2.tasks.lid import LIDTask
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.lid_trainer import LIDTrainer
from espnet2.train.reporter import Reporter


def _build_model(loss):
    """Build a LID model that mean-pools its input features."""
    return ESPnetLIDModel(
        frontend=None,
        specaug=None,
        normalize=None,
        encoder=IdentityEncoder(4),
        pooling=MeanPooling(4),
        projector=None,
        loss=loss,
    )


@pytest.mark.parametrize("custom_bs", [1, 3])
@pytest.mark.parametrize("extract_embd", [False, True])
@pytest.mark.parametrize(
    "loss_class",
    [
        ArcMarginProduct_intertopk_subcenter,
        ArcMarginProduct_intertopk_subcenter_softtarget,
        ArcMarginProduct_intertopk_subcenter_multilabel,
    ],
)
def test_lid_inference_target_modes(tmp_path, custom_bs, extract_embd, loss_class):
    """LID inference works with hard, soft and multi-label heads."""
    model = _build_model(loss_class(nout=4, nclasses=3, k_top=0))
    speech = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 5])
    if loss_class is ArcMarginProduct_intertopk_subcenter:
        labels = torch.tensor([[0], [1]])
        expected_counts = [1, 1, 0]
    else:
        labels = torch.tensor([[1.0, 0, 0], [0, 0.5, 0.5]])
        if loss_class is not ArcMarginProduct_intertopk_subcenter_softtarget:
            labels[1] *= 2
        expected_counts = [1, 1, 1]
    args = LIDTask.get_parser().parse_args([])
    args.ngpu = 0
    options = LIDTrainer.build_options(args)
    idx2lang = {0: "ara", 1: "eng", 2: "jpn"}
    embeddings = {lang: [] for lang in idx2lang.values()}
    counters = {lang: 0 for lang in idx2lang.values()}
    batch = {"speech": speech, "speech_lengths": lengths, "lid_labels": labels}
    report = Reporter()
    report.set_epoch(1)
    with report.observe("inference") as reporter:
        LIDTrainer.extract_embed_lid(
            model=model,
            iterator=iter([(["u1", "u2"], batch)]),
            reporter=reporter,
            options=options,
            distributed_option=DistributedOption(),
            output_dir=str(tmp_path),
            custom_bs=custom_bs,
            idx2lang=idx2lang,
            extract_embd=extract_embd,
            resume=False,
            lang_to_embds_dic=embeddings,
            max_num_utt_per_lang=1,
            lang_counter_dic=counters,
        )
    predictions = (tmp_path / "lids0").read_text().splitlines()
    assert len(predictions) == 2
    assert all(line.split()[1] in idx2lang.values() for line in predictions)
    assert list(counters.values()) == expected_counts
    if extract_embd:
        assert [len(values) for values in embeddings.values()] == expected_counts
        for values in embeddings.values():
            for embedding in values:
                np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


@pytest.mark.parametrize("custom_bs", [1, 3])
def test_lid_inference_without_labels_stores_per_utterance_embeddings(
    tmp_path, custom_bs
):
    """Per-utterance embeddings are saved even when no lid_labels are given."""
    model = _build_model(ArcMarginProduct_intertopk_subcenter(4, 3, k_top=0))
    args = LIDTask.get_parser().parse_args([])
    args.ngpu = 0
    options = LIDTrainer.build_options(args)
    idx2lang = {0: "ara", 1: "eng", 2: "jpn"}
    embeddings = {lang: [] for lang in idx2lang.values()}
    batch = {"speech": torch.randn(2, 5, 4), "speech_lengths": torch.tensor([5, 5])}
    report = Reporter()
    report.set_epoch(1)
    with report.observe("inference") as reporter:
        LIDTrainer.extract_embed_lid(
            model=model,
            iterator=iter([(["u1", "u2"], batch)]),
            reporter=reporter,
            options=options,
            distributed_option=DistributedOption(),
            output_dir=str(tmp_path),
            custom_bs=custom_bs,
            idx2lang=idx2lang,
            extract_embd=True,
            resume=False,
            lang_to_embds_dic=embeddings,
            save_embd_per_utt=True,
        )
    assert len((tmp_path / "lids0").read_text().splitlines()) == 2
    with np.load(tmp_path / "embeddings0.npz") as saved:
        assert sorted(saved) == ["u1", "u2"]
        for utt in saved:
            assert saved[utt].shape == (4,)
            np.testing.assert_allclose(np.linalg.norm(saved[utt]), 1.0, atol=1e-6)
    # grouping by language needs reference labels
    assert all(len(values) == 0 for values in embeddings.values())
